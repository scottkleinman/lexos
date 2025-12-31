"""d3_wordcloud.py.

Last Updated: August 12, 2025
Last Tested: December 5, 2025
"""

import json
import re
import tempfile
import webbrowser
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from jinja2 import Template
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    validate_call,
)
from smart_open import open
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc, Span, Token

from lexos.dtm import DTM
from lexos.exceptions import LexosException
from lexos.visualization import processors

# Valid input types
single_doc_types = dict[str, int] | Doc | Span | str | list[str] | list[Token]
multi_doc_types = (
    str
    | list[str]
    | list[list[str]]
    | list[Doc]
    | list[Span]
    | list[list[Token]]
    | dict[str, int]
    | pd.DataFrame
    | DTM
)


class D3WordCloud(BaseModel):
    """A Pydantic model for D3 WordCloud options."""

    data: single_doc_types | multi_doc_types | pd.DataFrame = Field(
        ...,
        description="The data to generate the word cloud from. Accepts data from a string, list of lists or tuples, a dict with terms as keys and counts/frequencies as values, or a dataframe.",
    )
    docs: Optional[int | str | list[int] | list[str]] = Field(
        None, description="A list of documents to be selected from the DTM."
    )
    layout: Optional[dict[str, Any]] = {}
    limit: int = Field(100, description="The maximum number of terms in the cloud.")
    font: str = Field("Impact", description="The font to use for the word cloud.")
    spiral: str = Field(
        "archimedean",
        description="The spiral type to use for the word cloud, 'archimedean' or 'rectangular'.",
    )
    scale: str = Field(
        "log",
        description="The scale type to use for the word cloud, 'log', 'sqrt', or 'linear'.",
    )
    angle_count: int = Field(
        5, description="The number of angles to use for the word cloud."
    )
    angle_from: int = Field(-60, description="The starting angle for the word cloud.")
    angle_to: int = Field(60, description="The ending angle for the word cloud.")
    width: int = Field(600, gt=50, description="The width of the word cloud in pixels.")
    height: int = Field(
        600, gt=50, description="The height of the word cloud in pixels."
    )
    title: str = Field(
        "Word Cloud Visualization", description="The title of the word cloud."
    )
    background_color: str = Field(
        "white", description="The background color of the word cloud."
    )
    colorscale: str = Field(
        "d3.scale.category20b",
        description="The name of a categorical d3 scale to use for the word cloud. See https://d3js.org/d3-scale.",
    )
    auto_open: bool = Field(
        True, description="Whether to open the chart in a web browser automatically."
    )
    template: Path | str = Field(
        "d3_cloud_template-1.0.html",
        description="The template file for the word cloud.",
    )
    auto_open: bool = Field(
        True, description="Whether to open the chart in a web browser automatically."
    )
    include_d3js: bool | str | None = Field(
        True,
        description="Whether to include the D3.js library. Can be 'cdn', 'directory', or a custom path. If False, the D3.js library will not be included. The cloud bundle is always included unless the setting is 'directory' or False.",
    )
    include_d3_cloud: bool | str = Field(
        True,
        description="Whether to include the D3 cloud library. Can be a custom path to a JavaScript file or True to use the default bundled version.",
    )
    counts: dict[str, int] = Field({}, description="A dictionary of word counts.")
    html: str = Field("", description="The HTML representation of the word cloud.")

    model_config = ConfigDict(
        arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
    )

    @field_validator("spiral")
    @classmethod
    def validate_spiral(cls, v):
        """Validate the spiral setting."""
        if v not in ["archimedean", "rectangular"]:
            raise ValueError('spiral must be "archimedean" or "rectangular"')
        return v

    @field_validator("scale")
    @classmethod
    def validate_scale(cls, v):
        """Validate the scale setting."""
        if v not in ["log", "sqrt", "linear"]:
            raise ValueError('scale must be "log", "sqrt", or "linear"')
        return v

    @model_validator(mode="after")
    def validate_angles(self):
        """Validate the angle settings."""
        if self.angle_from >= self.angle_to:
            raise ValueError("angle_from must be less than angle_to")
        return self

    def __init__(self, **data: Any) -> None:
        """Initialize with better error handling."""
        try:
            super().__init__(**data)
        except Exception as e:
            raise LexosException(f"Failed to initialize D3WordCloud: {e}") from e

        # Process the data into a consistent format
        self.counts = processors.process_data(self.data, self.docs, self.limit)
        self._render()
        self._include_d3()
        self._include_d3_cloud()

    def _load_template(self) -> str:
        """Load the HTML template for the word cloud."""
        template = self._get_asset_path(self.template)
        with open(template) as f:
            return f.read()

    def _render(self) -> None:
        """Render the word cloud as an HTML string."""
        template = Template(self._load_template())
        self.html = template.render(
            termCounts=json.dumps(self.counts),
            width=self.width,
            height=self.height,
            title=self.title,
            backgroundColor=self.background_color,
            colorscale=self.colorscale,
            font=self.font,
            spiral=self.spiral,
            scale=self.scale,
            angleCount=self.angle_count,
            angleFrom=self.angle_from,
            angleTo=self.angle_to,
        )

        # If auto_open is True, open the chart in a web browser
        if self.auto_open:
            self._open()

    def _get_asset_path(self, filename: str) -> Path:
        """Centralized asset path resolution."""
        return Path(__file__).parent / "d3_cloud_assets" / filename

    def _get_d3_js(self, path: str = "d3.min.js") -> str:
        """Retrieve the contents of the d3.js bundle.

        Args:
            path (str): The path to the d3.js file. Defaults to "d3.min.js".

        Returns:
            str: The HTML script tag containing or pointing to the d3.js script.
        """
        if path == "d3.min.js":
            path = self._get_asset_path("d3.min.js")
        try:
            with open(path) as f:
                return f'<script id="d3">\n{f.read()}\n</script>'
        except FileNotFoundError:
            raise LexosException(f"Script file not found: {path}")

    def _include_d3(self) -> None:
        """Modify the template to include d3.js and d3 cloud scripts."""
        # Handle loading/initializing d3.js
        if isinstance(self.include_d3js, str) and self.include_d3js.lower() == "cdn":
            load_d3js = (
                f'<script charset="utf-8" src="https://d3js.org/d3.min.js"></script>'
            )
        elif (
            isinstance(self.include_d3js, str)
            and self.include_d3js.lower() == "directory"
        ):
            load_d3js = f'<script charset="utf-8" src="{self._get_asset_path("d3.min.js")}"></script>'
        elif isinstance(self.include_d3js, str) and self.include_d3js.endswith(".js"):
            load_d3js = self._get_d3_js(self.include_d3js)
        elif self.include_d3js is True:
            load_d3js = self._get_d3_js()
        elif self.include_d3js is False:
            load_d3js = None  # Don't include d3.js
        elif self.include_d3js is None:
            load_d3js = self._get_d3_js()
        if load_d3js:
            self.html = self.html.replace('<script id="d3"></script>', load_d3js)

    def _include_d3_cloud(self) -> None:
        """Modify the template to include d3 cloud scripts."""
        # Handle loading/initializing d3 cloud
        if self.include_d3_cloud is True:
            path = f"{self._get_asset_path('d3cloud_bundle.min.js')}"
        elif isinstance(self.include_d3_cloud, str) and self.include_d3_cloud.endswith(
            ".js"
        ):
            path = self.include_d3_cloud

        if self.include_d3_cloud:
            with open(path) as f:
                self.html = self.html.replace(
                    '<script id="d3cloud"></script>',
                    f'<script id="d3cloud">\n{f.read()}\n</script>',
                )

    def _load_template(self) -> str:
        """Load the HTML template for the word cloud."""
        template = self._get_asset_path(self.template)
        with open(template) as f:
            return f.read()

    def _minify_html(self, html: str) -> str:
        """Basic HTML minification."""
        # Remove extra whitespace
        html = re.sub(r"\s+", " ", html)
        # Remove comments
        html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
        return html.strip()

    def _open(self) -> None:
        """Open the HTML file in a web browser."""
        # Create a temporary file to store the HTML
        with tempfile.NamedTemporaryFile(
            "w", delete=False, suffix=".html", encoding="utf-8"
        ) as temp_file:
            temp_file.write(self.html)
            temp_file_path = temp_file.name

            # Open the temporary HTML file in the default web browser
            webbrowser.open(f"file:///{temp_file_path}")

    @validate_call
    def save(self, path: Path | str, minify: bool = False) -> None:
        """Save the word cloud HTML to a file with optional HTML minification."""
        html_content = self.html

        if minify:
            html_content = self._minify_html(html_content)

        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)


class D3MultiCloud(BaseModel):
    """A Pydantic model for creating multiple D3 WordClouds in a grid layout."""

    data_sources: list[multi_doc_types] = Field(
        ...,
        description="List of data sources to create individual word clouds from.",
    )
    labels: Optional[list[str]] = Field(
        None,
        description="List of titles for each word cloud. If None, will use 'Cloud 1', 'Cloud 2', etc.",
    )
    cloud_width: int = Field(
        300, gt=50, description="The width of each individual word cloud in pixels."
    )
    cloud_height: int = Field(
        300, gt=50, description="The height of each individual word cloud in pixels."
    )
    columns: int = Field(
        3, gt=0, description="The number of columns in the grid layout."
    )
    title: Optional[str] = Field(None, description="Overall title for the figure.")
    cloud_spacing: int = Field(
        20, ge=0, description="The spacing between clouds in pixels."
    )
    limit: int = Field(50, description="The maximum number of terms in each cloud.")
    font: str = Field("Impact", description="The font to use for all word clouds.")
    spiral: str = Field(
        "archimedean",
        description="The spiral type to use for all word clouds.",
    )
    scale: str = Field(
        "log",
        description="The scale type to use for all word clouds.",
    )
    angle_count: int = Field(
        5, description="The number of angles to use for all word clouds."
    )
    angle_from: int = Field(-60, description="The starting angle for all word clouds.")
    angle_to: int = Field(60, description="The ending angle for all word clouds.")
    background_color: str = Field(
        "white", description="The background color of the overall visualization."
    )
    colorscale: str = Field(
        "d3.scale.category20b",
        description="The name of a categorical d3 scale to use for all word clouds.",
    )
    auto_open: bool = Field(
        True, description="Whether to open the chart in a web browser automatically."
    )
    template: Path | str = Field(
        "d3_multicloud_template-1.0.html",
        description="The template file for the multi-cloud visualization.",
    )
    include_d3js: bool | str | None = Field(
        True,
        description="Whether to include the D3.js library.",
    )
    include_d3_cloud: bool | str = Field(
        True,
        description="Whether to include the D3 cloud library.",
    )

    # Generated fields
    word_clouds: list[D3WordCloud] = Field(
        [], description="List of generated D3WordCloud objects."
    )
    html: str = Field("", description="The HTML representation of the multi-cloud.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("spiral")
    @classmethod
    def validate_spiral(cls, v):
        """Validate the spiral setting."""
        if v not in ["archimedean", "rectangular"]:
            raise LexosException('spiral must be "archimedean" or "rectangular"')
        return v

    @field_validator("scale")
    @classmethod
    def validate_scale(cls, v):
        """Validate the scale setting."""
        if v not in ["log", "sqrt", "linear"]:
            raise LexosException('scale must be "log", "sqrt", or "linear"')
        return v

    @model_validator(mode="after")
    def validate_angles(self):
        """Validate the angle settings."""
        if self.angle_from >= self.angle_to:
            raise ValueError("angle_from must be less than angle_to")
        return self

    def __init__(self, **data: Any) -> None:
        """Initialize the multi-cloud visualization."""
        try:
            super().__init__(**data)
        except Exception as e:
            raise LexosException(f"Failed to initialize D3MultiCloud: {e}") from e

        # Generate labels if not provided
        if self.labels is None:
            self.labels = [f"Doc {i + 1}" for i in range(len(self.data_sources))]
        elif len(self.labels) != len(self.data_sources):
            raise LexosException(
                "Number of labels must match number of data sources or be None"
            )

        # Generate individual word clouds
        self._generate_word_clouds()

        # Generate the combined HTML
        self._render()

    def _generate_word_clouds(self) -> None:
        """Generate individual D3WordCloud objects for each data source."""
        self.word_clouds = []

        for i, (data_source, label) in enumerate(zip(self.data_sources, self.labels)):
            cloud = D3WordCloud(
                data=data_source,
                width=self.cloud_width,
                height=self.cloud_height,
                title=label,
                limit=self.limit,
                font=self.font,
                spiral=self.spiral,
                scale=self.scale,
                angle_count=self.angle_count,
                angle_from=self.angle_from,
                angle_to=self.angle_to,
                background_color=self.background_color,
                colorscale=self.colorscale,
                auto_open=False,
                include_d3js=False,  # We'll include D3 once in the master template
                include_d3_cloud=False,  # We'll include cloud lib once in the master template
            )
            self.word_clouds.append(cloud)

    def _get_asset_path(self, filename: str) -> Path:
        """Centralized asset path resolution."""
        return Path(__file__).parent / "d3_cloud_assets" / filename

    def _get_cloud(self, index: int) -> D3WordCloud:
        """Get a specific word cloud by index."""
        if 0 <= index < len(self.word_clouds):
            return self.word_clouds[index]
        raise IndexError(f"Cloud index {index} out of range")

    def _get_d3_js(self, path: str = "d3.min.js") -> str:
        """Retrieve the contents of the d3.js bundle."""
        if path == "d3.min.js":
            path = self._get_asset_path("d3.min.js")
        try:
            with open(path) as f:
                return f'<script id="d3">\n{f.read()}\n</script>'
        except FileNotFoundError:
            raise LexosException(f"Script file not found: {path}")

    def _include_d3(self) -> None:
        """Modify the template to include d3.js."""
        if isinstance(self.include_d3js, str) and self.include_d3js.lower() == "cdn":
            load_d3js = (
                '<script charset="utf-8" src="https://d3js.org/d3.v3.min.js"></script>'
            )
        elif (
            isinstance(self.include_d3js, str)
            and self.include_d3js.lower() == "directory"
        ):
            load_d3js = f'<script charset="utf-8" src="{self._get_asset_path("d3.min.js")}"></script>'
        elif isinstance(self.include_d3js, str) and self.include_d3js.endswith(".js"):
            load_d3js = self._get_d3_js(self.include_d3js)
        elif self.include_d3js is True:
            load_d3js = self._get_d3_js()
        elif self.include_d3js is False:
            load_d3js = ""
        else:
            load_d3js = self._get_d3_js()

        self.html = self.html.replace('<script id="d3"></script>', load_d3js)

    def _include_d3_cloud(self) -> None:
        """Modify the template to include d3 cloud scripts."""
        if self.include_d3_cloud is True:
            path = "d3_cloud_assets/d3cloud_bundle.min.js"
        elif isinstance(self.include_d3_cloud, str) and self.include_d3_cloud.endswith(
            ".js"
        ):
            path = self.include_d3_cloud
        else:
            path = "d3_cloud_assets/d3cloud_bundle.min.js"

        if self.include_d3_cloud:
            try:
                with open(path) as f:
                    self.html = self.html.replace(
                        '<script id="d3cloud"></script>',
                        f'<script id="d3cloud">\n{f.read()}\n</script>',
                    )
            except FileNotFoundError:
                # Fallback to CDN
                self.html = self.html.replace(
                    '<script id="d3cloud"></script>',
                    '<script src="https://cdn.jsdelivr.net/gh/jasondavies/d3-cloud/build/d3.layout.cloud.js"></script>',
                )

    def _load_template(self) -> str:
        """Load the HTML template for the multi-cloud visualization."""
        template = self._get_asset_path(self.template)
        with open(template) as f:
            return f.read()

    def _open(self) -> None:
        """Open the HTML file in a web browser."""
        # Create a temporary file to store the HTML
        with tempfile.NamedTemporaryFile(
            "w", delete=False, suffix=".html", encoding="utf-8"
        ) as temp_file:
            temp_file.write(self.html)
            temp_file_path = temp_file.name

            # Open the temporary HTML file in the default web browser
            webbrowser.open(f"file:///{temp_file_path}")

    def _render(self) -> None:
        """Generate the combined HTML for all word clouds."""
        template = Template(self._load_template())

        # Calculate grid dimensions
        rows = (len(self.word_clouds) + self.columns - 1) // self.columns
        total_width = (self.cloud_width * self.columns) + (
            self.cloud_spacing * (self.columns - 1)
        )
        total_height = (self.cloud_height * rows) + (self.cloud_spacing * (rows - 1))

        # Prepare cloud data for template
        cloud_data = []
        for i, cloud in enumerate(self.word_clouds):
            row = i // self.columns
            col = i % self.columns
            x_pos = col * (self.cloud_width + self.cloud_spacing)
            y_pos = row * (self.cloud_height + self.cloud_spacing)

            cloud_data.append(
                {
                    "id": f"cloud_{i}",
                    "title": cloud.title,
                    "termCounts": cloud.counts,
                    "x": x_pos,
                    "y": y_pos,
                    "width": self.cloud_width,
                    "height": self.cloud_height,
                }
            )

        self.html = template.render(
            title=self.title,
            total_width=total_width,
            total_height=total_height + 100,  # Extra space for title
            cloud_data=json.dumps(cloud_data),
            font=self.font,
            spiral=self.spiral,
            scale=self.scale,
            angleCount=self.angle_count,
            angleFrom=self.angle_from,
            angleTo=self.angle_to,
            backgroundColor=self.background_color,
            colorscale=self.colorscale,
        )

        # Include D3 libraries
        self._include_d3()
        self._include_d3_cloud()

        # If auto_open is True, open the chart in a web browser
        if self.auto_open:
            self._open()

    @validate_call
    def get_cloud_counts(self, index: int) -> dict[str, int]:
        """Get word counts for a specific cloud by index."""
        return self._get_cloud(index).counts

    @validate_call
    def save(self, path: Path | str, minify: bool = False) -> None:
        """Save the multi-cloud HTML to a file."""
        html_content = self.html

        if minify:
            import re

            html_content = re.sub(r"\s+", " ", html_content)
            html_content = re.sub(r"<!--.*?-->", "", html_content, flags=re.DOTALL)
            html_content = html_content.strip()

        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)
