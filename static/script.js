$(document).ready(function() {
    // Load locations
    $.get('/locations', function(locations) {
        Object.keys(locations).forEach(key => {
            const loc = locations[key];
            $('#locationSelect').append(
                `<option value="${key}">${loc.name}</option>`
            );
        });
    });

    // Handle location selection
    $('#locationSelect').change(function() {
        const selectedKey = $(this).val();
        if (selectedKey) {
            $.get('/locations', function(locations) {
                const coords = locations[selectedKey].coordinates;
                $('#lat_min').val(coords.lat_min);
                $('#lon_min').val(coords.lon_min);
                $('#lat_max').val(coords.lat_max);
                $('#lon_max').val(coords.lon_max);
            });
        }
    });

    $('#analysisForm').on('submit', function(e) {
        e.preventDefault();
        
        const data = {
            lat_min: parseFloat($('#lat_min').val()),
            lon_min: parseFloat($('#lon_min').val()),
            lat_max: parseFloat($('#lat_max').val()),
            lon_max: parseFloat($('#lon_max').val()),
            start_date: $('#start_date').val(),
            end_date: $('#end_date').val()
        };

        // Show loading
        $('#results').hide();
        $('#loading').show();

        // Send request
        $.ajax({
            url: '/analyze',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(data),
            success: function(response) {
                $('#loading').hide();
                $('#results').show();
                
                // Update image and report
                $('#resultImage').attr('src', response.image_url);
                
                // Load report text
                $.get(response.report_url, function(report) {
                    $('#report').text(report);
                });
            },
            error: function(xhr) {
                $('#loading').hide();
                alert('Error: ' + xhr.responseJSON.error);
            }
        });
    });
});
