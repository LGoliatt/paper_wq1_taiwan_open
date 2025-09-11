import pandas as pd
import folium
from folium.plugins import MarkerCluster
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import contextily as ctx
from io import BytesIO
import requests
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patches as mpatches

# Step 1: Read the CSV file from Google Sheets
key = '1a5DReajqstsnUSUdTcRm8pZqeIP9ZmOct834UcOLmjg'
link = 'https://docs.google.com/spreadsheet/ccc?key=' + key + '&output=csv'
r = requests.get(link)
data = r.content
df = pd.read_csv(BytesIO(data), header=0)

# Step 2: Clean and prepare data
# Remove duplicates based on siteid, sitename, and sampledate
df = df.drop_duplicates(subset=['siteid', 'sitename', 'sampledate'])

# Filter relevant columns
df = df[['siteid', 'sitename', 'siteengname', 'county', 'basin', 'river', 'twd97lon', 'twd97lat', 'sampledate', 'itemengabbreviation', 'itemvalue', 'itemunit']]

# Remove duplicates based on coordinates (twd97lon, twd97lat)
unique_sites = df.drop_duplicates(subset=['twd97lon', 'twd97lat'])

# Convert longitude and latitude to float
unique_sites['twd97lon'] = pd.to_numeric(unique_sites['twd97lon'], errors='coerce')
unique_sites['twd97lat'] = pd.to_numeric(unique_sites['twd97lat'], errors='coerce')
unique_sites = unique_sites.dropna(subset=['twd97lon', 'twd97lat'])

# Step 3: Create a professional interactive map using Folium
# Center the map on Taiwan (approximate center: lat 23.7, lon 121.0)
taiwan_map = folium.Map(location=[23.7, 121.0], zoom_start=7, tiles='cartodbpositron')

# Add marker cluster for better visualization
marker_cluster = MarkerCluster().add_to(taiwan_map)

# Add points to the map
for idx, row in unique_sites.iterrows():
    popup_text = f"""
    <b>Site ID:</b> {row['siteid']}<br>
    <b>Site Name:</b> {row['sitename']} ({row['siteengname']})<br>
    <b>County:</b> {row['county']}<br>
    <b>Basin/River:</b> {row['basin']} / {row['river']}<br>
    <b>Latest Sample Date:</b> {row.get('sampledate', 'N/A')}<br>
    """
    site_data = df[df['siteid'] == row['siteid']]
    rpi_row = site_data[site_data['itemengabbreviation'] == 'RPI']
    if not rpi_row.empty:
        rpi_value = float(rpi_row['itemvalue'].values[0])
        popup_text += f"<b>RPI:</b> {rpi_value}<br>"
        color = 'green' if rpi_value < 2 else 'orange' if rpi_value < 4 else 'red'
    else:
        color = 'blue'
    
    folium.Marker(
        location=[row['twd97lat'], row['twd97lon']],
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(color=color, icon='cloud')
    ).add_to(marker_cluster)

# Add layer control
folium.LayerControl().add_to(taiwan_map)

# Save the interactive map
taiwan_map.save('taiwan_water_quality_map_interactive.html')
print("Interactive map saved as 'taiwan_water_quality_map_interactive.html'")

# Step 4: Create a static professional map using GeoPandas and Matplotlib
# Convert to GeoDataFrame
geometry = [Point(lon, lat) for lon, lat in zip(unique_sites['twd97lon'], unique_sites['twd97lat'])]
gdf = gpd.GeoDataFrame(unique_sites, geometry=geometry, crs="EPSG:4326")

# Project to Web Mercator (EPSG:3857) for basemap
gdf = gdf.to_crs(epsg=3857)

# Plot the map
fig, ax = plt.subplots(figsize=(10, 12))
gdf.plot(ax=ax, markersize=50, color='red', edgecolor='black', alpha=0.7, label='Monitoring Sites')

# Add basemap (Taiwan outline)
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=7)

# Set bounds to focus on Taiwan
#ax.set_xlim(13300000, 13600000)
#ax.set_ylim(2500000, 3000000)

# Add title and labels
#ax.set_title('Taiwan Water Quality Monitoring Sites', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Longitude (EPSG:3857)', fontsize=12)
ax.set_ylabel('Latitude (EPSG:3857)', fontsize=12)

# Add scale bar (in kilometers, bottom-left corner)
scale = ScaleBar(dx=1, units="km", location='lower left', scale_loc='bottom', length_fraction=0.2)
ax.add_artist(scale)

# Add north arrow (top-right corner)
x, y, arrow_length = 0.95, 0.15, 0.1
ax.annotate('N', xy=(x, y), xytext=(x, y - arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center', va='center', fontsize=12, xycoords=ax.transAxes)
ax.text(x, y + 0.02, 'North', transform=ax.transAxes, ha='center', va='bottom', fontsize=10)

# Add grid and legend
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper left', fontsize=10)

# Tight layout and save as high-resolution PNG
plt.tight_layout()
plt.savefig('taiwan_water_quality_map_static.png', dpi=300, bbox_inches='tight')
print("Static map saved as 'taiwan_water_quality_map_static.png'")
