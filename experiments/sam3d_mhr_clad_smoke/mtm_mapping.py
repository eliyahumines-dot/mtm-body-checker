"""Mapping from clad-body's ISO 8559-1 measurement keys to MTM tailoring
terminology.

This is a terminology map, not a claim of definitional equivalence. ISO
8559-1 (garment sizing survey methodology) and bespoke/MTM tailoring each
define "chest," "waist," etc. with their own landmark conventions, and
those conventions do not always coincide (e.g. a tailor's chest measurement
is typically taken over the fullest part with the tape held per house
convention and garment-ease assumptions; ISO 8559-1's bust/chest definition
follows a body-scan-derived landmark protocol). Where we know of a specific
divergence we say so in ``note``; where we don't, ``note`` says so rather
than asserting silent equivalence, per Task 02's explicit instruction not
to conflate ISO and tailoring-house definitions.

``confidence`` is our own qualitative judgement about how directly usable
the clad-body value is for MTM purposes as-is, not an accuracy claim:
- "direct": names and landmark intent are close enough to use as a first
  pass without adjustment.
- "review": usable but the tailoring convention commonly differs enough
  (ease allowance, exact landmark) that a human should sanity-check it
  before treating a discrepancy flag as reliable.
- "gap": no clad-body/ISO key covers this MTM measurement at all.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MeasurementMapping:
    mtm_name: str
    clad_body_key: str | None
    iso_ref: str | None
    note: str
    confidence: str  # "direct" | "review" | "gap"


MTM_MEASUREMENT_MAP: tuple[MeasurementMapping, ...] = (
    MeasurementMapping(
        mtm_name="height",
        clad_body_key="height_cm",
        iso_ref=None,
        note="Vertical mesh extent, feet to crown. Matches standing height "
             "conventions closely; least likely of any measurement here to "
             "carry a definitional mismatch.",
        confidence="direct",
    ),
    MeasurementMapping(
        mtm_name="chest / bust circumference",
        clad_body_key="bust_cm",
        iso_ref="5.3.4",
        note="ISO 'bust' is measured at the fullest bust/chest point per "
             "ISO 8559-1 landmark convention. Tailoring-house chest "
             "measurement is usually taken with the tape level around the "
             "fullest part under the arms, often with a stated ease "
             "allowance for the garment -- clad-body reports the bare-body "
             "circumference, not ease-adjusted. Treat any comparison to a "
             "customer-reported chest measurement as 'bare body vs. "
             "customer's own tape technique', not an apples-to-apples "
             "match.",
        confidence="review",
    ),
    MeasurementMapping(
        mtm_name="waist circumference",
        clad_body_key="waist_cm",
        iso_ref=None,
        note="clad-body computes waist as an 'exact' extent measurement "
             "(same loop/extent method as height, per its own docs), not a "
             "soft-argmin convex-hull circumference like bust/hip. Where "
             "exactly ISO/clad-body's waist landmark sits (natural waist? "
             "narrowest point?) vs. a tailor's stated waist landmark is "
             "UNVERIFIED -- flag for review before trusting a discrepancy "
             "here.",
        confidence="review",
    ),
    MeasurementMapping(
        mtm_name="seat / hip circumference",
        clad_body_key="hip_cm",
        iso_ref="5.3.4 area (hip)",
        note="ISO hip landmark (fullest part of seat/hip) is generally "
             "close to tailoring 'seat' convention, but exact landmark "
             "height and convex-hull-vs-contour tape technique can differ "
             "by a few cm on curvier bodies (clad-body's own docs note the "
             "convex-hull vs. raw-contour gap is 'most visible ... on "
             "larger cup sizes' for bust; a similar effect is plausible at "
             "the seat for some body shapes -- UNVERIFIED how large).",
        confidence="review",
    ),
    MeasurementMapping(
        mtm_name="shoulder width",
        clad_body_key="shoulder_width_cm",
        iso_ref=None,
        note="clad-body's own published calibration error for this key is "
             "the largest of any reported: RMS 1.39 cm on 100 random "
             "bodies (91% within +/-2 cm, max 5 cm) -- and that is against "
             "clad-body's own internal reference, not real humans. "
             "Shoulder-point landmark placement varies significantly "
             "between tailoring houses (acromion vs. visually-judged "
             "shoulder seam point). Treat as the least reliable length "
             "measurement in this list even before considering SAM 3D "
             "Body's own prediction error.",
        confidence="review",
    ),
    MeasurementMapping(
        mtm_name="sleeve / arm length",
        clad_body_key="sleeve_length_cm",
        iso_ref=None,
        note="clad-body's 'sleeve_length_cm' models a straight sleeve "
             "pattern length, not necessarily a tailor's 'sleeve length' "
             "which is often measured shoulder-point to wrist with the arm "
             "slightly bent, following the arm's natural drape. Different "
             "houses define this differently even among tailors -- "
             "UNVERIFIED whether clad-body's convention matches ours.",
        confidence="review",
    ),
    MeasurementMapping(
        mtm_name="upper-arm / bicep circumference",
        clad_body_key="upperarm_cm",
        iso_ref=None,
        note="Reasonably direct concept match (fullest part of upper arm), "
             "clad-body's own published error vs. its internal reference "
             "is tight (<= 1 cm).",
        confidence="direct",
    ),
    MeasurementMapping(
        mtm_name="neck circumference",
        clad_body_key="neck_cm",
        iso_ref="5.3.2",
        note="ISO 8559-1 section 5.3.2 explicitly defines the landmark as "
             "'just below the Adam's apple,' which clad-body's docs cite "
             "directly -- this is one of the more explicitly-documented "
             "landmark definitions in the library. Tailoring collar-size "
             "convention is usually close to this but can include a small "
             "fixed ease allowance depending on house convention.",
        confidence="review",
    ),
    MeasurementMapping(
        mtm_name="back length",
        clad_body_key=None,
        iso_ref=None,
        note="No clad-body key found for 'back length' (nape-to-waist along "
             "the back) as a standalone measurement. clad-body has a "
             "'back-neck-to-waist' style length per its README's mention of "
             "'shirt length' and related back lengths, but we have not "
             "confirmed an exact key name or landmark match -- treat as a "
             "gap pending direct verification against clad-body's live "
             "REGISTRY.",
        confidence="gap",
    ),
    MeasurementMapping(
        mtm_name="front torso length",
        clad_body_key=None,
        iso_ref=None,
        note="No confirmed clad-body key for a front-specific torso length "
             "(as distinct from back length or overall shirt length). Gap "
             "pending direct verification against clad-body's live "
             "REGISTRY.",
        confidence="gap",
    ),
    MeasurementMapping(
        mtm_name="wrist circumference",
        clad_body_key="wrist_cm",
        iso_ref=None,
        note="clad-body's README lists 'wrist' among its 25 measurements; "
             "we have not independently pulled its published calibration "
             "error for this key in this task -- treat accuracy as "
             "unverified even though the key exists.",
        confidence="review",
    ),
    MeasurementMapping(
        mtm_name="inseam",
        clad_body_key="inseam_cm",
        iso_ref=None,
        note="clad-body's own published calibration error is tight (RMS "
             "0.06 cm, max 0.10 cm) against its internal reference. Crotch "
             "landmark placement can still differ from a tailor's inseam "
             "convention (measured on an existing well-fitting garment, "
             "often, rather than on the body directly) -- different "
             "measurement *method*, not just different landmark, so treat "
             "any direct comparison with caution regardless of clad-body's "
             "internal precision.",
        confidence="review",
    ),
    MeasurementMapping(
        mtm_name="outseam",
        clad_body_key=None,
        iso_ref=None,
        note="No confirmed clad-body key for outseam (waist to floor along "
             "the outside leg) as a standalone measurement -- gap pending "
             "direct verification against clad-body's live REGISTRY. "
             "clad-body's README mentions 'front/back rise' which may be "
             "related but is not confirmed equivalent.",
        confidence="gap",
    ),
    MeasurementMapping(
        mtm_name="thigh circumference",
        clad_body_key="thigh_cm",
        iso_ref=None,
        note="clad-body's own published calibration error is tight (MAE "
             "0.06 cm, max 0.18 cm on 100 random bodies) against its "
             "internal reference. Landmark height (upper thigh vs. a "
             "specific measured distance below crotch) not independently "
             "confirmed against tailoring convention.",
        confidence="review",
    ),
)


def as_lookup() -> dict[str, MeasurementMapping]:
    return {m.mtm_name: m for m in MTM_MEASUREMENT_MAP}
