# Cutter

`Cutter` is a module that divides files, texts, or documents into segments. At present, it is highly experimental, and only the methods for splitting spaCy documents (pre-tokenised texts) is available. These methods are implemented through two classes: `Ginsu` and `Machete`.

`Ginsu` acts as a more precise cutter, allowing for greater accuracy in exchange for longer loading times.

`Machete`, on the other hand, allows for faster loading in exchange for precision.

<iframe style="width: 560px; height: 315px; margin: auto;" src="https://www.youtube.com/embed/Sv_uL1Ar0oM" title="YouTube video player -- Ginsu knives" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### ::: lexos.cutter.Ginsu
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.cutter.Machete
    rendering:
      show_root_heading: true
      heading_level: 3