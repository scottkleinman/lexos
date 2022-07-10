/* Load the Metadata into the Modal */
$(document).on("click", ".external", function (e) {
  e.preventDefault();
  var url = $(this).attr("href");
  let doc_id = url.split("/").pop();
  let doc_zero_id = doc_id - 1;
  var data;
  $.ajax({
    type: "GET",
    url: "data/meta.csv",
    dataType: "text",
    success: function (response) {
      data = $.csv.toArrays(response);
      let row = data[doc_id];
      let out = "ID: " + doc_id + "<br>";
      out += "Title: " + row[1] + "<br>";
      out += "Author: " + row[2] + "<br>";
      out += "Publication: " + row[3] + "<br>";
      if (row[4] != "") {
        out += "Volume: " + row[5] + "<br>";
      }
      if (row[4] != "") {
        out += "Volume: " + row[4] + "<br>";
      }
      out += "Year: " + row[6] + "<br>";
      if (row[7] != "") {
        out += "Page Range: " + row[7] + "<br>";
      }
      $("#modalMeta").html(out);
      $.get("data/docs/" + doc_zero_id + ".txt", function (data) {
        // Truncate after first 500 characters
        const truncate = (data) =>
          data.length > 500 ? `${data.substring(0, 500)}…` : data;
        $("#modalText").html(truncate(data));
        let button =
          '<div style="margin-top: 10px;"><a id="' +
          doc_zero_id +
          '" data-title="' +
          row[1] +
          '" role="button" class="btn btn-sm btn-default url fulltext" href="#">View Full Text</a></div>';
        $("#modalText").append(button);
      });
      $("#jsonDoc").modal();
      $("#loading").hide();
    },
    fail: function () {
      alert("Failed to load meta.csv");
    },
  });
});

$(document).on("click", ".fulltext", function (e) {
  e.preventDefault();
  let id = $(this).attr("id");
  var title = $(this).attr("data-title");
  $.get("data/docs/" + id + ".txt", function (data) {
    var newWin = open("", title);
    newWin.document.write(data);
    newWin.document.close();
  });
});

/* Handle the check box to show punctuation and spaces */
$(document).on("change", "#show-punct", function () {
  if ($(this).prop("checked") == true) {
    // Remove the tabel filter
    $("#featureTable").bootstrapTable("filterBy", {});
  } else {
    // Get the data and re-create the POS list without PUNCT and SPACE
    let data = $("#featureTable").bootstrapTable("getData");
    let pos_tags = [];
    $.each(data, function (i, v) {
      pos_tags.push(v["pos"]);
    });
    pos_tags = pos_tags.filter(function (value) {
      return value !== "PUNCT" && value !== "SPACE";
    });
    // Filter the table
    $("#featureTable").bootstrapTable("filterBy", {
      pos: pos_tags,
    });
  }
});

/* Handle tooltip labels */
$(document).on("mouseenter", ".tv-top-word-label", function () {
  $(this).tooltip("show");
});
globalMouseEventHandler();
