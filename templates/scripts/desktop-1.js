document.addEventListener("DOMContentLoaded", function() {
  // Initialize the map
  var map = new google.maps.Map(document.getElementById("map-container"), {
    center: { lat: 0, lng: 0 }, // Set the initial center of the map
    zoom: 12, // Adjust the zoom level according to your preference
  });
});