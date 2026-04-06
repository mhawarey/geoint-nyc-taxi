# NYC Taxi Geospatial Intelligence System

**Advanced Trajectory Clustering & Real-Time Mobility Analytics**

A comprehensive geospatial intelligence platform that processes millions of NYC yellow taxi trip records to discover hidden mobility patterns, optimize route planning, and provide actionable business insights through machine learning clustering and interactive visualization.

## Key Results

| Metric | Value |
|--------|-------|
| Records Processed | 9.4 million taxi trips |
| Time Span | 3 months (Jan–Mar 2023) |
| Mobility Clusters Discovered | 248 unique patterns |
| Signal-to-Noise Ratio | 87.8% |
| Peak Usage Pattern | 5–6 PM rush hour concentration |

## Technical Approach

- **Spatial Clustering**: DBSCAN algorithm with haversine distance metric for geographic coordinate clustering — no assumption on cluster count required
- **Feature Engineering**: Pickup/dropoff GPS coordinates, trip duration, distance, temporal features (hour, day-of-week), fare decomposition
- **Scalable Pipeline**: Efficient processing of compressed Parquet files with chunked loading and memory-aware sampling
- **Visualization**: Publication-quality geospatial dashboards using Plotly and Matplotlib/Seaborn

## Architecture

```
geoint.py          # Core analysis engine: data loading, preprocessing, DBSCAN clustering, visualization
geoint2.py         # Extended analysis variant with additional clustering configurations
data/              # NYC TLC yellow taxi trip data (Parquet format, not included — see below)
```

## Dependencies

```
pandas
numpy
scikit-learn
plotly
matplotlib
seaborn
pyarrow          # Parquet file support
```

## Data

This project uses NYC Taxi & Limousine Commission (TLC) yellow taxi trip data:
- Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- Format: `.parquet` (compressed)
- Files used: `yellow_tripdata_2023-01.parquet`, `yellow_tripdata_2023-02.parquet`, `yellow_tripdata_2023-03.parquet`

> **Note**: Raw data files (~145 MB compressed) are not included in this repository. Download from the TLC website above and place in the project root directory.

## Usage

```bash
pip install pandas numpy scikit-learn plotly matplotlib seaborn pyarrow
python geoint.py
```

The system will automatically detect `.parquet.zip` files in the working directory, extract, preprocess, cluster, and generate interactive visualizations.

## Methodology

1. **Data Ingestion**: Parallel extraction and loading of compressed Parquet archives
2. **Preprocessing**: Outlier removal (invalid GPS coordinates, extreme durations/distances), feature scaling with `StandardScaler`
3. **DBSCAN Clustering**: Density-based spatial clustering using haversine distances — discovers arbitrarily shaped clusters without requiring predefined cluster count
4. **Temporal Analysis**: Hour-of-day and day-of-week pattern extraction across clusters
5. **Geospatial Visualization**: Heatmaps, cluster maps, temporal distribution charts, corridor analysis

## Author

**Dr. Mosab Hawarey**
PhD, Geodetic & Photogrammetric Engineering | MSc, Geomatics (Purdue) | MSc, Geodesy (METU)

- GitHub: [github.com/mhawarey](https://github.com/mhawarey)
- ORCID: [0000-0001-7846-951X](https://orcid.org/0000-0001-7846-951X)

## License

MIT License
