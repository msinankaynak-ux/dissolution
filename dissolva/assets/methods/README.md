# Dissolution method figures — drop-in slot

The Academy "Dissolution Methods" chapter shows an original SVG schematic for each
method by default. To use a purchased/licensed image instead, drop a file here named
by the method key:

    assets/methods/<key>.png   (or .jpg / .svg / .webp)

If a file exists for a key, the app shows it automatically (with the credit caption);
otherwise it falls back to the built-in schematic. Keys:

    usp1_basket, usp2_paddle, usp3_recip_cylinder, usp4_flow_cell,
    usp5_paddle_disk, usp6_cylinder, usp7_recip_holder,
    franz_cell, dialysis_bag, reverse_dialysis, sample_separate, continuous_flow

Only add images you have the right to use (purchased license, CC-BY with attribution,
public domain, or your own). Put the credit text in methods_content.py -> "figure_credit".
