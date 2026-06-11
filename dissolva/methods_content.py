"""DissolvA Academy — Dissolution Methods chapter content.

Each method: name, suitable dosage forms, operating conditions, description,
literature refs, and an ORIGINAL SVG schematic (copyright-clean). To use a
purchased/licensed image instead, drop a file at assets/methods/<key>.<ext>
(see assets/methods/README.md) and set "figure_credit" below for attribution.
"""

METHOD_ORDER = [
    "usp1_basket", "usp2_paddle", "usp3_recip_cylinder", "usp4_flow_cell",
    "usp5_paddle_disk", "usp6_cylinder", "usp7_recip_holder",
    "franz_cell", "dialysis_bag", "reverse_dialysis", "sample_separate",
    "continuous_flow",
]

_S = '<svg viewBox="0 0 300 230" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;max-height:260px;">'
_VES = 'fill="none" stroke="#aebfda" stroke-width="2"'
_LIQ = 'fill="rgba(120,160,230,0.18)"'
_MET = 'stroke="#cfd8ea" stroke-width="2" fill="none"'
_LBL = 'fill="#9fb0d0" font-family="sans-serif" font-size="11"'
_AMB = 'fill="#FFCC00"'

METHODS = {
 "usp1_basket": {
  "name": "USP Apparatus I — Rotating Basket",
  "dosage_forms": "Tablets and capsules — especially formulations that float or stick to the vessel bottom in the paddle apparatus.",
  "conditions": "Typically 900 mL medium at 37 ± 0.5 °C; basket rotation 50–100 rpm; sinkers not needed (basket retains the unit).",
  "description": "The dosage unit is enclosed in a cylindrical wire-mesh basket attached to a rotating shaft and immersed in the medium. Rotation creates the hydrodynamic flow past the unit. Apparatus I is preferred when the dosage form would otherwise float or adhere to the vessel.",
  "refs": ["USP General Chapter <711> Dissolution", "FDA (1997) Guidance for Industry: Dissolution Testing of IR Solid Oral Dosage Forms"],
  "figure_credit": "",
  "svg": _S + '<path d="M70 40 Q60 40 60 55 L60 175 Q60 205 95 205 L205 205 Q240 205 240 175 L240 55 Q240 40 230 40" '+_VES+'/>'
        '<path d="M62 120 L62 175 Q62 203 95 203 L205 203 Q238 203 238 175 L238 120 Z" '+_LIQ+'/>'
        '<line x1="150" y1="18" x2="150" y2="120" '+_MET+'/>'
        '<rect x="128" y="120" width="44" height="55" rx="3" fill="none" stroke="#cfd8ea" stroke-width="1.5" stroke-dasharray="2 3"/>'
        '<ellipse cx="150" cy="148" rx="11" ry="7" '+_AMB+'/>'
        '<text x="150" y="14" text-anchor="middle" '+_LBL+'>rotating shaft</text>'
        '<text x="250" y="150" '+_LBL+'>mesh</text><text x="250" y="163" '+_LBL+'>basket</text>'
        '<text x="186" y="150" '+_AMB+' font-size="10" font-family="sans-serif">unit</text>'
        '<text x="20" y="120" '+_LBL+'>medium</text>'
        '</svg>',
 },
 "usp2_paddle": {
  "name": "USP Apparatus II — Paddle",
  "dosage_forms": "The default for immediate- and modified-release tablets and capsules (sinkers used for floating units).",
  "conditions": "Typically 900 mL (500–1000 mL) at 37 ± 0.5 °C; paddle 50–75 rpm; Q usually evaluated at 30–45 min for IR.",
  "description": "A flat paddle on a rotating shaft stirs the medium; the dosage unit rests at the bottom of the round-bottom vessel (a sinker keeps floating units submerged). The most widely used apparatus for oral solids worldwide.",
  "refs": ["USP General Chapter <711> Dissolution", "FDA (1997) IR Dissolution Guidance", "Stevens, Gray & Dorantes (2015) AAPS J 17(2):301-306"],
  "figure_credit": "",
  "svg": _S + '<path d="M70 40 Q60 40 60 55 L60 175 Q60 205 95 205 L205 205 Q240 205 240 175 L240 55 Q240 40 230 40" '+_VES+'/>'
        '<path d="M62 95 L62 175 Q62 203 95 203 L205 203 Q238 203 238 175 L238 95 Z" '+_LIQ+'/>'
        '<line x1="150" y1="18" x2="150" y2="165" '+_MET+'/>'
        '<path d="M120 165 Q150 185 180 165 L176 175 Q150 192 124 175 Z" fill="#cfd8ea"/>'
        '<ellipse cx="150" cy="196" rx="12" ry="6" '+_AMB+'/>'
        '<text x="150" y="14" text-anchor="middle" '+_LBL+'>rotating shaft</text>'
        '<text x="186" y="160" '+_LBL+'>paddle</text>'
        '<text x="168" y="200" '+_AMB+' font-size="10" font-family="sans-serif">tablet</text>'
        '<text x="20" y="110" '+_LBL+'>medium</text>'
        '</svg>',
 },
 "usp3_recip_cylinder": {
  "name": "USP Apparatus III — Reciprocating Cylinder (Bio-Dis)",
  "dosage_forms": "Modified/extended-release and multi-pH studies; beads, pellets, chewables — units moved through a row of media to mimic GI transit.",
  "conditions": "Inner glass cylinder (mesh top & bottom) reciprocates vertically (e.g., 5–35 dpm) through a row of vessels at 37 °C, each with a different medium/pH.",
  "description": "The dosage form sits in an inner cylinder that dips up and down (reciprocates) inside outer tubes. By moving the unit sequentially through vessels of changing pH, it reproduces the pH/time profile of GI transit — useful for ER dosage-form development.",
  "refs": ["USP General Chapter <711> Dissolution", "Samaha, Shehayeb & Kyriacos (2009) Dissolution Technologies 16(2):41-46"],
  "figure_credit": "",
  "svg": _S + '<rect x="55" y="70" width="50" height="135" rx="4" '+_VES+'/>'
        '<rect x="125" y="70" width="50" height="135" rx="4" '+_VES+'/>'
        '<rect x="195" y="70" width="50" height="135" rx="4" '+_VES+'/>'
        '<rect x="57" y="120" width="46" height="83" '+_LIQ+'/><rect x="127" y="120" width="46" height="83" '+_LIQ+'/><rect x="197" y="120" width="46" height="83" '+_LIQ+'/>'
        '<rect x="64" y="55" width="32" height="60" rx="3" fill="none" stroke="#cfd8ea" stroke-width="1.5"/>'
        '<ellipse cx="80" cy="92" rx="8" ry="5" '+_AMB+'/>'
        '<line x1="80" y1="30" x2="80" y2="55" '+_MET+'/>'
        '<path d="M80 22 l5 9 h-10 z" '+_AMB+'/><path d="M80 40 l-5 -9 h10 z" '+_AMB+'/>'
        '<text x="150" y="225" text-anchor="middle" '+_LBL+'>row of vessels — changing pH (GI transit)</text>'
        '<text x="100" y="50" '+_LBL+'>reciprocating cylinder</text>'
        '</svg>',
 },
 "usp4_flow_cell": {
  "name": "USP Apparatus IV — Flow-Through Cell",
  "dosage_forms": "Low-solubility drugs, modified release, implants, suspensions, powders, microparticulates — open or closed loop with fresh medium.",
  "conditions": "Medium pumped upward through a vertical cell (with glass beads as flow distributor) at a set flow rate (e.g., 4–16 mL/min) at 37 °C; sink conditions easily maintained.",
  "description": "A piston pump drives medium upward through a cell holding the dosage form on a bed of glass beads, then to a reservoir or fraction collector. Because fresh medium continuously contacts the unit, true sink conditions are easy to maintain — ideal for poorly soluble drugs and special dosage forms.",
  "refs": ["USP General Chapter <711> Dissolution", "Costa & Sousa Lobo (2001) Eur J Pharm Sci 13(2):123-133"],
  "figure_credit": "",
  "svg": _S + '<path d="M120 60 L180 60 L168 150 Q150 175 132 150 Z" '+_VES+'/>'
        '<path d="M124 95 L176 95 L166 148 Q150 170 134 148 Z" '+_LIQ+'/>'
        '<circle cx="150" cy="150" r="3" fill="#cfd8ea"/><circle cx="143" cy="156" r="3" fill="#cfd8ea"/><circle cx="157" cy="156" r="3" fill="#cfd8ea"/>'
        '<ellipse cx="150" cy="110" rx="10" ry="7" '+_AMB+'/>'
        '<line x1="150" y1="180" x2="150" y2="205" '+_MET+'/><line x1="150" y1="60" x2="150" y2="40" '+_MET+'/>'
        '<path d="M150 196 l5 -9 h-10 z" fill="#5dd0ff"/><path d="M150 50 l5 -9 h-10 z" fill="#5dd0ff"/>'
        '<rect x="40" y="190" width="40" height="22" rx="3" '+_VES+'/><text x="60" y="205" text-anchor="middle" '+_LBL+'>pump</text>'
        '<line x1="80" y1="201" x2="135" y2="201" '+_MET+'/>'
        '<text x="190" y="100" '+_LBL+'>flow-through</text><text x="190" y="113" '+_LBL+'>cell</text>'
        '<text x="186" y="113" '+_AMB+' font-size="10" font-family="sans-serif"></text>'
        '<text x="160" y="35" '+_LBL+'>to reservoir</text>'
        '</svg>',
 },
 "usp5_paddle_disk": {
  "name": "USP Apparatus V — Paddle over Disk",
  "dosage_forms": "Transdermal patches and topical systems — the patch is held flat, release face up.",
  "conditions": "Paddle apparatus with a horizontal disk assembly holding the patch at the vessel bottom; 37 ± 0.5 °C (often 32 °C for skin-surface relevance); defined rpm.",
  "description": "A flat disk assembly holds a transdermal system release-side up at the bottom of a standard paddle vessel; the paddle stirs above it. Provides a fixed exposed area and reproducible hydrodynamics for patch release testing.",
  "refs": ["USP General Chapter <711> Dissolution", "USP <724> Drug Release"],
  "figure_credit": "",
  "svg": _S + '<path d="M70 40 Q60 40 60 55 L60 175 Q60 205 95 205 L205 205 Q240 205 240 175 L240 55 Q240 40 230 40" '+_VES+'/>'
        '<path d="M62 95 L62 175 Q62 203 95 203 L205 203 Q238 203 238 175 L238 95 Z" '+_LIQ+'/>'
        '<line x1="150" y1="18" x2="150" y2="150" '+_MET+'/>'
        '<path d="M122 150 Q150 168 178 150 L174 159 Q150 175 126 159 Z" fill="#cfd8ea"/>'
        '<rect x="116" y="186" width="68" height="12" rx="3" fill="none" stroke="#cfd8ea" stroke-width="2"/>'
        '<rect x="128" y="182" width="44" height="6" '+_AMB+'/>'
        '<text x="150" y="14" text-anchor="middle" '+_LBL+'>shaft + paddle</text>'
        '<text x="188" y="190" '+_LBL+'>disk +</text><text x="188" y="202" '+_AMB+' font-size="10" font-family="sans-serif">patch</text>'
        '</svg>',
 },
 "usp6_cylinder": {
  "name": "USP Apparatus VI — Cylinder",
  "dosage_forms": "Transdermal systems mounted on the outer surface of a rotating cylinder.",
  "conditions": "A stainless cylinder replaces the basket/paddle; the transdermal system is fixed to its outside; immersed in medium at 37 °C (often 32 °C), defined rpm.",
  "description": "Derived from Apparatus I, the wire basket is replaced by a solid cylinder. The transdermal patch is mounted on the cylinder's exterior and rotated in the medium, giving a controlled exposed area for release testing of patches.",
  "refs": ["USP General Chapter <711> Dissolution", "USP <724> Drug Release"],
  "figure_credit": "",
  "svg": _S + '<path d="M70 40 Q60 40 60 55 L60 175 Q60 205 95 205 L205 205 Q240 205 240 175 L240 55 Q240 40 230 40" '+_VES+'/>'
        '<path d="M62 95 L62 175 Q62 203 95 203 L205 203 Q238 203 238 175 L238 95 Z" '+_LIQ+'/>'
        '<line x1="150" y1="18" x2="150" y2="110" '+_MET+'/>'
        '<rect x="128" y="110" width="44" height="70" rx="4" fill="none" stroke="#cfd8ea" stroke-width="2"/>'
        '<rect x="124" y="124" width="6" height="42" '+_AMB+'/><rect x="170" y="124" width="6" height="42" '+_AMB+'/>'
        '<text x="150" y="14" text-anchor="middle" '+_LBL+'>rotating shaft</text>'
        '<text x="186" y="120" '+_LBL+'>cylinder</text>'
        '<text x="60" y="120" '+_AMB+' font-size="10" font-family="sans-serif">patch on surface</text>'
        '</svg>',
 },
 "usp7_recip_holder": {
  "name": "USP Apparatus VII — Reciprocating Holder",
  "dosage_forms": "Transdermal systems, drug-eluting stents, implants and other small/non-disintegrating dosage forms.",
  "conditions": "Sample holders on a rod reciprocate vertically into a row of small-volume tubes (low media volumes) at 37 °C; useful for low-dose / extended studies.",
  "description": "Dosage forms are attached to holders on a vertically reciprocating rod that dips into a series of small tubes. The small media volumes and gentle motion suit transdermals, stents and implants studied over extended times.",
  "refs": ["USP General Chapter <711> Dissolution", "USP <724> Drug Release"],
  "figure_credit": "",
  "svg": _S + '<rect x="60" y="120" width="34" height="85" rx="4" '+_VES+'/><rect x="133" y="120" width="34" height="85" rx="4" '+_VES+'/><rect x="206" y="120" width="34" height="85" rx="4" '+_VES+'/>'
        '<rect x="62" y="150" width="30" height="53" '+_LIQ+'/><rect x="135" y="150" width="30" height="53" '+_LIQ+'/><rect x="208" y="150" width="30" height="53" '+_LIQ+'/>'
        '<line x1="60" y1="60" x2="240" y2="60" '+_MET+'/>'
        '<line x1="77" y1="60" x2="77" y2="160" '+_MET+'/><line x1="150" y1="60" x2="150" y2="160" '+_MET+'/><line x1="223" y1="60" x2="223" y2="160" '+_MET+'/>'
        '<rect x="71" y="150" width="12" height="14" '+_AMB+'/><rect x="144" y="150" width="12" height="14" '+_AMB+'/><rect x="217" y="150" width="12" height="14" '+_AMB+'/>'
        '<line x1="150" y1="30" x2="150" y2="60" '+_MET+'/><path d="M150 24 l5 9 h-10 z" '+_AMB+'/><path d="M150 44 l-5 -9 h10 z" '+_AMB+'/>'
        '<text x="150" y="220" text-anchor="middle" '+_LBL+'>holders reciprocate into small tubes</text>'
        '</svg>',
 },
}
