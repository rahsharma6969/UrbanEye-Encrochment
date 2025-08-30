-- Enable PostGIS extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_raster;
CREATE EXTENSION IF NOT EXISTS uuid-ossp;

-- Table for Areas of Interest (AOI)
CREATE TABLE IF NOT EXISTS aoi (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    geom GEOMETRY(MULTIPOLYGON, 4326) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Table for Detections (encroachments / change polygons)
CREATE TABLE IF NOT EXISTS detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    aoi_id UUID NOT NULL REFERENCES aoi(id) ON DELETE CASCADE,
    t0 TIMESTAMPTZ NOT NULL,   -- before date range
    t1 TIMESTAMPTZ NOT NULL,   -- after date range
    bbox GEOMETRY(POLYGON, 4326), -- bounding box of change area
    mask_url TEXT,             -- path to raster/COG or GeoJSON mask
    score REAL,                -- confidence score (0–1)
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_aoi_geom ON aoi USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_detections_bbox ON detections USING GIST (bbox);
CREATE INDEX IF NOT EXISTS idx_detections_aoi_id ON detections(aoi_id);

