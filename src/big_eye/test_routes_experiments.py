import folium
import numpy as np

center = [31.83, 117.23]
map_1 = folium.Map(location=center,
           tiles='Stamen Toner',
           zoom_start=13)
line_points = []
for _ in range(10):
    point = [x + np.random.normal(0, 0.01) for x in center]
    line_points.append(point)
    folium.Marker(point, icon=folium.Icon(color='red')).add_to(map_1)
folium.PolyLine(line_points).add_to(map_1)
map_1.show