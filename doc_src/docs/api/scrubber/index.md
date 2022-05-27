# Scrubber

`Scrubber` is a destructive preprocessing module that contains a set of functions for manipulating text. It leans heavily on the code base for <a href="https://github.com/chartbeat-labs/textacy/" target="_blank">Textacy</a> but tweaks some of that library's functions in order to modify or extend the functionality.

`Scrubber` is divided into five submodules:

<table>
    <colgroup>
        <col style="width: 30%">
        <col style="width: 70%">
    </colgroup>
    <tbody>
        <tr class="row-odd">
            <td><a href="https://scottkleinman.github.io/lexos/api/scrubber/normalize/" title="lexos.scrubber.normalize"><code>normalize</code></a></td>
            <td>A set of functions for massaging text into standardised forms.</td>
        </tr>
        <tr class="row-even">
            <td><a href=".https://scottkleinman.github.io/lexos/api/scrubber/pipeline/" title="lexos.scrubber.pipeline"><code>pipeline</code></a></td>
            <td>A set of functions for feeding multiple components into a
            scrubbing function.</td>
        </tr>
        <tr class="row-odd">
            <td><a href="https://scottkleinman.github.io/lexos/api/scrubber/registry/" title="lexos.scrubber.registry"><code>registry</code></a></td>
            <td>A registry of scrubbing functions that can be accessed to
            reference functions by name.</td>
        </tr>
        <tr class="row-even">
            <td><a href="https://scottkleinman.github.io/lexos/api/scrubber/remove" title="lexos.scrubber.remove"><code>remove</code></a></td>
            <td>A set of functions for removing strings and patterns from text.</td>
        </tr>
        <tr class="row-odd">
            <td><a href="https://scottkleinman.github.io/lexos/api/scrubber/replace/" title="lexos.scrubber.replace"><code>replace</code></a></td>
            <td>A set of functions for replacing strings and patterns from text.</td>
        </tr>
        <tr class="row-even">
            <td><a href="https://scottkleinman.github.io/lexos/api/scrubber/resources/" title="lexos.scrubber.resources"><code>resources</code></a></td>
            <td>A set of constants, classes, and functions used by the other components of the <code>Scrubber</code> module.</td>
        </tr>
        <tr class="row-odd">
            <td><a href="https://scottkleinman.github.io/lexos/api/scrubber/scrubber/" title="lexos.scrubber.scrubber"><code>scrubber</code></a></td>
            <td>Constains the <code>lexos.scrubber.scrubber.Scrub</code> class for managing scrubbing pipelines.</td>
        </tr>
        <tr class="row-even">
            <td><a href="https://scottkleinman.github.io/lexos/api/scrubber/utils/" title="lexos.scrubber.utils"><code>utils</code></a></td>
            <td>A set of utility functions shared by the other components of the <code>Scrubber</code> module.</td>
        </tr>
    </tbody>
</table>
