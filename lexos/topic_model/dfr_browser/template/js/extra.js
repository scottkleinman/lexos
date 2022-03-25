/* Load the Metadata into the Modal */
$(document).on('click', '.external', function (e) {
    e.preventDefault()
    let url = $(this).attr('href')
    $.getJSON(url, function (result) {
            $('#loading').show()
            var data = []
            var pos_tags = []
            var terms = []
            // Convert the array to an object with features and counts
            if (result.hasOwnProperty('features')) {
                $.each(result['features'], function (index, col) {
                    terms.push({'features': col, 'count': 0})
                    /* Deprecated code for showing the feature table
                        if (index !== 0) {
                            let cols = {
                                'index': index, 'token': col[0], 'norm': col[1], 'lemma': col[2],
                                'pos': col[3], 'tag': col[4], 'stopword': col[5], 'entities': col[6]
                            }
                            data.push(cols)
                        }
                    */
                })

                // Reduce the array to unique features and add counts for each feature
                let term_counts = []
                terms.forEach(el => {
                    let entry = term_counts.find(x => x.features.join(',') === el.features.join(','))
                    if (entry !== undefined) {
                        entry.count = entry.count + 1
                    } else {
                        term_counts.push({'features': el.features, 'count': 1})
                    }
                })

                // Create the table columns and record the unique POS labels used in the data
                $.each(term_counts, function (index, val) {
                    col = val.features
                    pos_tags.push(col[3])
                    let cols = {
                        'index': index,
                        'token': col[0],
                        'norm': col[1],
                        'lemma': col[2],
                        'pos': col[3],
                        'tag': col[4],
                        'stopword': col[5],
                        'entities': col[6],
                        'count': val.count
                    }
                    data.push(cols)
                })

                // Remove PUNCT and SPACE from the POS labels
                pos_tags = [...new Set(pos_tags)]
                pos_tags = pos_tags.filter(function (value) {
                    return value !== 'PUNCT' && value !== 'SPACE'
                })

                // Constants for the jsonMetadata div
                let spacy_msg = 'WE1S uses spaCy\'s annotation scheme for <a href="https://spacy.io/api/annotation#pos-tagging" target="_blank" style="color: #007bff!important;">parts of speech</a>, and <a href="https://spacy.io/api/annotation#named-entities" target="_blank" style="color: #007bff!important;">named entity</a> tags.'
            } else if (result.hasOwnProperty('bag_of_words')) {
                $.each(result['bag_of_words'], function (term, count) {
                    data.push({'term': term, 'count': count})
                })
            }
        
            let metadata = ''
            // Build the html for each metadata field
            $.each(result, function (index, value) {
                if (index !== 'content' && index !== 'content_scrubbed') {
                    // If features are present, build html table headers and add the message about spaCy
                    if (index === 'features') {
                        let table = '<div class="toolbar"><label><input type="checkbox" id="show-punct" /> Show punctuation and spaces</label></div><table id="featureTable" class="table-sm table-striped" data-pagination="true" data-sortable="true" data-toolbar=".toolbar"><thead><tr>'
                        table += '<td data-field="index"></td><th data-field="token" data-sortable="true">TOKEN</th><th data-field="norm" data-sortable="true">NORM</th><th data-field="lemma" data-sortable="true">LEMMA</th><th data-field="pos" data-sortable="true">POS</th><th data-field="tag" data-sortable="true">TAG</th><th data-field="stopword" data-sortable="true">STOPWORD</th><th data-field="entities" data-sortable="true">ENTITIES</th><th data-field="count" data-sortable="true">COUNT</th>'
                        table += '</tr></thead>'
                        table += '<tbody></table>'
                        metadata += '<p><strong>' + index + ':</strong><br>' + spacy_msg + '</p>' + table
                    // Otherwise, just add the metadata item
                    } else if (index === 'bag_of_words') {
                        let table = '<table id="bowTable" class="table-sm table-striped" data-pagination="true" data-sortable="true"><thead><tr>'
                        table += '<td data-field="index"></td><th data-field="term" data-sortable="true">TERM</th><th data-field="count" data-sortable="true">COUNT</th>'
                        table += '</tr></thead>'
                        table += '<tbody></table>'
                        metadata += '<p><strong>bag_of_words:</strong></p>' + table
                    } else {
                        metadata += '<strong>' + index + ':</strong> ' + value + '<br>'
                    }
                }
            })

            // Insert the metadata into the DOM and initialise the Bootstrap Table
            $('#jsonMetadata').html(metadata)
            if (result.hasOwnProperty('features')) {
                $('#featureTable').bootstrapTable({
                    data: data
                })
                // Filter the table so that PUNCT and SPACE are not shown
                $('#featureTable').bootstrapTable('filterBy', {
                    'pos': pos_tags
                })
            } else if (result.hasOwnProperty('bag_of_words')) {
                $('#bowTable').bootstrapTable({
                    data: data
                })
            }
            // Add the link to the json file
            $('#newWindowLink').attr('href', url)

            // Remove the content section and header
            $('#jsonMetadata').next().remove()
            $('#jsonContent').remove()

            // Open the modal and hide the loading message
            $('#jsonDoc').modal()
            $('#loading').hide()
        })
        .fail(function () {
            alert('You do not have access to this file or the file could not be found.')
        })
})

/* Handle the check box to show punctuation and spaces */
$(document).on('change', '#show-punct', function () {
    if ($(this).prop('checked') == true) {
        // Remove the tabel filter
        $('#featureTable').bootstrapTable('filterBy', {})
    } else {
        // Get the data and re-create the POS list without PUNCT and SPACE
        let data = $('#featureTable').bootstrapTable('getData')
        let pos_tags = []
        $.each(data, function (i, v) {
            pos_tags.push(v['pos'])
        })
        pos_tags = pos_tags.filter(function (value) {
            return value !== 'PUNCT' && value !== 'SPACE'
        })
        // Filter the table
        $('#featureTable').bootstrapTable('filterBy', {
            'pos': pos_tags
        })
    }
})

/* Handle tooltip labels */
$(document).on('mouseenter', '.tv-top-word-label', function () {
    $(this).tooltip('show')
})
globalMouseEventHandler()