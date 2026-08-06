# Examples

BRAHMS is used entirely through its graphical interface — there are no
Python scripts to edit. This page walks through a typical session, tab by
tab.

## 1. Nonlinear Crystals

Define the crystal you want to simulate directly from its Sellmeier
equation(s), specifying whether it is quasi-phase-matched (QPM) or
birefringent, together with its relevant parameters ($d_\mathrm{eff}$,
linear and two-photon absorption, walk-off angle, thermal properties,
grating period for QPM crystals, etc.). Common crystals (e.g., MgO:PPLN,
MgO:sPPLT, PPLN, ZGP) are pre-loaded with literature Sellmeier
coefficients; you can add, edit, or remove custom crystals here.

## 2. Phase Matching

Given the selected crystal and a set of experimental parameters (pump
wavelength, temperature, polarization, and, for QPM crystals, grating
period), this tab computes the phase-matching condition ($\Delta k$ as a
function of the relevant parameter) for the three-wave-mixing process you
want to study (SHG, SFG, or OPG).

## 3. Single simulation

Runs one (3+1)D or reduced-dimensionality simulation for a given
configuration, on either the CPU or GPU backend. Use this tab to explore a
single set of physical parameters and inspect the resulting beam profiles,
phases, and (optionally) thermal profile.

## 4. Parameter Sweep

Automates repeated simulations while scanning one physical parameter
(e.g., pump power, beam waist, crystal temperature, or phase mismatch) to
compute conversion-efficiency curves and identify optimal operating
conditions.

## 5. (3+1)D Simulations

Runs the full space- and time-resolved simulation in `focused-pulsed`
mode — the most demanding configuration BRAHMS supports, and the one that
benefits the most from the GPU backend.

---

For the underlying physical model and numerical method, see the paper (in
preparation) or the [README](./README.md#package-description).
