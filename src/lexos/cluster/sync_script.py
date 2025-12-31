"""sync_script.py.

This script synchronizes the heatmap and dendrogram axes in a Plotly clustermap.
It is added to the HTML output of the clustermap to ensure that when the user zooms or pans on one axis, the corresponding axes are updated accordingly.

Last Updated: 25 July, 2025
"""

SYNC_SCRIPT = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        var plotDiv = document.querySelector('.plotly-graph-div');
        if (plotDiv) {
            var isUpdating = false;

            // Store original ranges and mapping information
            var originalRanges = {
                heatmapX: null,
                heatmapY: null,
                colDendX: null,
                rowDendY: null
            };

            // Store the number of data points for proper scaling
            var dataInfo = {
                numRows: 50,  // Number of rows in heatmap
                numCols: 4    // Number of columns in heatmap
            };

            // Initialize original ranges
            setTimeout(function() {
                var layout = plotDiv.layout;
                originalRanges.heatmapX = [0, dataInfo.numCols - 1];
                originalRanges.heatmapY = [0, dataInfo.numRows - 1];
                originalRanges.colDendX = layout.xaxis2.range || [0, 45];
                originalRanges.rowDendY = layout.yaxis3.range || [0, 500];
            }, 100);

            plotDiv.on('plotly_relayout', function(eventData) {
                if (isUpdating) return;
                isUpdating = true;

                var updates = {};
                var hasUpdates = false;

                // Handle heatmap x-axis changes
                if (eventData['xaxis4.range[0]'] !== undefined) {
                    var heatmapXRange = [eventData['xaxis4.range[0]'], eventData['xaxis4.range[1]']];

                    // Calculate proportional range for column dendrogram
                    var heatmapWidth = originalRanges.heatmapX[1] - originalRanges.heatmapX[0];
                    var dendXWidth = originalRanges.colDendX[1] - originalRanges.colDendX[0];

                    var startProp = (heatmapXRange[0] - originalRanges.heatmapX[0]) / heatmapWidth;
                    var endProp = (heatmapXRange[1] - originalRanges.heatmapX[0]) / heatmapWidth;

                    var dendXStart = originalRanges.colDendX[0] + (startProp * dendXWidth);
                    var dendXEnd = originalRanges.colDendX[0] + (endProp * dendXWidth);

                    updates['xaxis2.range'] = [dendXStart, dendXEnd];
                    hasUpdates = true;
                }

                // Handle heatmap y-axis changes
                if (eventData['yaxis4.range[0]'] !== undefined) {
                    var heatmapYRange = [eventData['yaxis4.range[0]'], eventData['yaxis4.range[1]']];

                    // Calculate proportional range for row dendrogram
                    var heatmapHeight = originalRanges.heatmapY[1] - originalRanges.heatmapY[0];
                    var dendYHeight = originalRanges.rowDendY[1] - originalRanges.rowDendY[0];

                    var startProp = (heatmapYRange[0] - originalRanges.heatmapY[0]) / heatmapHeight;
                    var endProp = (heatmapYRange[1] - originalRanges.heatmapY[0]) / heatmapHeight;

                    var dendYStart = originalRanges.rowDendY[0] + (startProp * dendYHeight);
                    var dendYEnd = originalRanges.rowDendY[0] + (endProp * dendYHeight);

                    updates['yaxis3.range'] = [dendYStart, dendYEnd];
                    hasUpdates = true;
                }

                // Handle column dendrogram changes
                if (eventData['xaxis2.range[0]'] !== undefined) {
                    var dendXRange = [eventData['xaxis2.range[0]'], eventData['xaxis2.range[1]']];

                    var dendXWidth = originalRanges.colDendX[1] - originalRanges.colDendX[0];
                    var heatmapWidth = originalRanges.heatmapX[1] - originalRanges.heatmapX[0];

                    var startProp = (dendXRange[0] - originalRanges.colDendX[0]) / dendXWidth;
                    var endProp = (dendXRange[1] - originalRanges.colDendX[0]) / dendXWidth;

                    var heatmapXStart = originalRanges.heatmapX[0] + (startProp * heatmapWidth);
                    var heatmapXEnd = originalRanges.heatmapX[0] + (endProp * heatmapWidth);

                    updates['xaxis4.range'] = [heatmapXStart, heatmapXEnd];
                    hasUpdates = true;
                }

                // Handle row dendrogram changes - FIXED VERSION
                if (eventData['yaxis3.range[0]'] !== undefined) {
                    var dendYRange = [eventData['yaxis3.range[0]'], eventData['yaxis3.range[1]']];

                    var dendYHeight = originalRanges.rowDendY[1] - originalRanges.rowDendY[0];
                    var heatmapHeight = originalRanges.heatmapY[1] - originalRanges.heatmapY[0];

                    // Calculate proportion based on the dendrogram's coordinate system
                    var startProp = (dendYRange[0] - originalRanges.rowDendY[0]) / dendYHeight;
                    var endProp = (dendYRange[1] - originalRanges.rowDendY[0]) / dendYHeight;

                    // Map to heatmap coordinates, but consider the reversed y-axis
                    // The dendrogram y-axis is NOT reversed, but heatmap y-axis IS reversed
                    var heatmapYStart = originalRanges.heatmapY[0] + ((1 - endProp) * heatmapHeight);
                    var heatmapYEnd = originalRanges.heatmapY[0] + ((1 - startProp) * heatmapHeight);

                    // Ensure we don't go outside bounds
                    heatmapYStart = Math.max(originalRanges.heatmapY[0], Math.min(originalRanges.heatmapY[1], heatmapYStart));
                    heatmapYEnd = Math.max(originalRanges.heatmapY[0], Math.min(originalRanges.heatmapY[1], heatmapYEnd));

                    updates['yaxis4.range'] = [heatmapYStart, heatmapYEnd];
                    hasUpdates = true;
                }

                if (hasUpdates) {
                    Plotly.relayout(plotDiv, updates).then(function() {
                        isUpdating = false;
                    });
                } else {
                    isUpdating = false;
                }
            });

            // Handle selection events specifically
            plotDiv.on('plotly_selected', function(eventData) {
                if (isUpdating || !eventData || !eventData.range) return;
                isUpdating = true;

                var updates = {};
                var range = eventData.range;

                // Handle row dendrogram selection (yaxis3)
                if (range.y && eventData.points && eventData.points[0] && eventData.points[0].yaxis === 'y3') {
                    var dendYRange = [range.y[0], range.y[1]];

                    var dendYHeight = originalRanges.rowDendY[1] - originalRanges.rowDendY[0];
                    var heatmapHeight = originalRanges.heatmapY[1] - originalRanges.heatmapY[0];

                    var startProp = (dendYRange[0] - originalRanges.rowDendY[0]) / dendYHeight;
                    var endProp = (dendYRange[1] - originalRanges.rowDendY[0]) / dendYHeight;

                    // Handle reversed y-axis mapping
                    var heatmapYStart = originalRanges.heatmapY[0] + ((1 - endProp) * heatmapHeight);
                    var heatmapYEnd = originalRanges.heatmapY[0] + ((1 - startProp) * heatmapHeight);

                    updates['yaxis3.range'] = dendYRange;
                    updates['yaxis4.range'] = [heatmapYStart, heatmapYEnd];

                    Plotly.relayout(plotDiv, updates).then(function() {
                        isUpdating = false;
                    });
                } else {
                    isUpdating = false;
                }
            });
        }
    });
    </script>
"""
