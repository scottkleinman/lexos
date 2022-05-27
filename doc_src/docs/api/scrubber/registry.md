# Registry

The `registry` component of `Scrubber` maintains
a catalogue of registered functions that can be imported individually as
needed. The registry enables the functions to be referenced by name using
string values. The code registry is created and accessed using the <a href="https://github.com/explosion/catalogue" target="_blank">catalogue</a> library by Explosion.

Registered functions can be retrieved individually using `lower_case = scrubber_components.get("lower_case")`. Multiple functions can be loaded using the `load_components` function:

### ::: lexos.scrubber.registry.load_component
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.scrubber.registry.load_components
    rendering:
      show_root_heading: true
      heading_level: 3

!!! note

    Custom functions can be registered by first creating the
    function and then adding it to the registry. An example is given below:

    ```python
    from lexos.scrubber.registry import scrubber_components

    def title_case(text):
        """Convert text to title case using `title()`"""
        return text.title()

    scrubber_components.register("title_case", func=title_case)
    ```
