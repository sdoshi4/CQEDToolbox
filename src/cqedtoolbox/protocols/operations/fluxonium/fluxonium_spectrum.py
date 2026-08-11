"""Fluxonium + resonator spectrum, as an lmfit model.

Fits the coupling ``g`` and the bare resonator frequency ``fr`` to a measured
dressed resonator frequency f_r(flux), holding EJ/EC/EL fixed.  Those three are
measured separately (qubit spectroscopy near zero and half flux), so here they
are inputs, not free parameters.

Ported from `resonator_fit_function.ipynb` in the KCCharacterization notebooks,
retargeted from plottr's Fit base class to labcore's -- the two share the
model/guess/run API.

``scqubits`` is imported at call time rather than at module import, so this
module can be imported on machines that only run measurements.  Install it with
the ``theory`` extra.
"""

import logging

import numpy as np

from labcore.analysis.fit import Fit

logger = logging.getLogger(__name__)


def _scq():
    """Import scqubits on demand, with an actionable error if it is absent."""
    try:
        import scqubits
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "scqubits is required for the fluxonium theory fit. "
            "Install it with: pip install 'cqedtoolbox[theory]'"
        ) from exc
    return scqubits


class FluxoniumResonatorFit(Fit):
    """Dressed resonator frequency vs external flux.

    coordinates
        External flux in units of the flux quantum.
    data
        Measured dressed resonator frequency, GHz.

    EJ/EC/EL are set on the instance before running, because lmfit only varies
    what ``model`` declares as parameters::

        fit = FluxoniumResonatorFit(flux, fr_measured)
        fit.EJ, fit.EC, fit.EL = 3.38815, 1.14658, 0.50108
        result = fit.run()
    """

    #: Fluxonium charge-basis cutoff and retained levels.
    CUTOFF = 120
    QUBIT_LEVELS = 10
    #: Resonator Fock levels kept in the joint space. The notebook left this at
    #: the scqubits default; pinning it makes cost and accuracy reproducible.
    RESONATOR_LEVELS = 4

    # Set before run(); class-level fallbacks match the notebook's globals.
    EJ = 3.38815
    EC = 1.14658
    EL = 0.50108

    def model(self, coordinates: np.ndarray, g: float, fr: float) -> np.ndarray:
        """Dressed |g,1> - |g,0> transition at each flux point."""
        scq = _scq()
        transitions = []

        for flux in np.atleast_1d(coordinates):
            fluxonium = scq.Fluxonium(
                EJ=self.EJ,
                EC=self.EC,
                EL=self.EL,
                cutoff=self.CUTOFF,
                flux=float(flux),
                truncated_dim=self.QUBIT_LEVELS,
            )
            resonator = scq.Oscillator(
                E_osc=fr,
                truncated_dim=self.RESONATOR_LEVELS,
            )

            hilbertspace = scq.HilbertSpace([fluxonium, resonator])
            hilbertspace.add_interaction(
                # The -1j carries the overall factor of the n * (a - a^dagger)
                # coupling; the Hermitian conjugate is already inside the
                # operator difference, so it must not be added again.
                g=g * -1j,
                op1=(fluxonium.n_operator, fluxonium),
                op2=((resonator.annihilation_operator()
                      - resonator.creation_operator()), resonator),
                add_hc=False,
            )
            hilbertspace.generate_lookup()

            g0 = hilbertspace.energy_by_bare_index((0, 0))
            g1 = hilbertspace.energy_by_bare_index((0, 1))
            transitions.append(g1 - g0)

        return np.asarray(transitions, dtype=float)

    @staticmethod
    def guess(coordinates, data):
        """Seed from the data: fr near the measured mean, g at a typical 60 MHz.

        The coupling cannot be read off the curve directly -- it sets the size
        of the avoided crossing -- so it keeps the notebook's starting value.
        """
        data = np.asarray(data, dtype=float)
        fr = float(np.mean(data)) if data.size else 6.463
        return dict(g=60e-3, fr=fr)
