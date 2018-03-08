import psycopg2

'''
This program loads the 'features' and supporting 'dnn_streams' tables into a postgres database named video-query.
Currently it works for a database of that name created locally by user torres.
The SQL CREATE scripts are copied from pgAdmin 4.
For now, we are assuming the deep net used to compute features is a TSN net trained on UCF-101 data, 
and the ensemble averaging is over models trained on splits 1, 2, and 3.
'''

conn = psycopg2.connect("host=localhost dbname=video-query user=torres")  # creates Connection object
cur = conn.cursor()   # creates Cursor object for issuing commands

# create dnn_streams table, a simple table with allowed names of stream/ensemble number options
cur.execute("""
    CREATE TABLE public.dnn_streams
    (
        type character varying(80) COLLATE pg_catalog."default" NOT NULL,
        max_number_of_splits smallint,
        CONSTRAINT dnn_streams_dnn_stream_unique UNIQUE (type)
    )
    WITH (
        OIDS = FALSE
    )
    TABLESPACE pg_default;
    
    COMMENT ON COLUMN public.dnn_streams.max_number_of_splits
        IS 'maximum allowed number of splits for this stream';
""")
# add names of stream/ensemble options
cur.execute("""
    INSERT INTO dnn_streams (type, max_number_of_splits)
    VALUES (%s, %s), (%s, %s), (%s, %s);
    """,
            ['rgb', 3, 'optical flow', 3, 'warped_optical_flow', 3]
)

# create video table
cur.execute("""
    CREATE TABLE public.video
    (
        id BIGSERIAL PRIMARY KEY,
        name character varying(254) COLLATE pg_catalog."default" NOT NULL,
        path character varying(4096),
        CONSTRAINT video_name_unique UNIQUE (name)
    )
    WITH (
        OIDS = FALSE
    )
    TABLESPACE pg_default;
""")

# create video_clips table
cur.execute("""
    CREATE TABLE public.video_clips
    (
        id BIGSERIAL PRIMARY KEY,
        video_id BIGINT NOT NULL,
        clip integer NOT NULL,
        duration integer NOT NULL DEFAULT 10,
        debug_video_uri character varying(4096) COLLATE pg_catalog."default",
        notes character varying(25600) COLLATE pg_catalog."default",
        CONSTRAINT no_duplicate_clips UNIQUE (video_id, clip, duration),
        CONSTRAINT positive_clip_duration CHECK (duration > 0)
    )
    WITH (
        OIDS = FALSE
    )
    TABLESPACE pg_default;
    
    ALTER TABLE public.video_clips
        OWNER to torres;
    COMMENT ON TABLE public.video_clips
        IS 'clips of videos for searching';
    
    COMMENT ON COLUMN public.video_clips.video_id
        IS 'id (primary key) of video in the video table';
    
    COMMENT ON COLUMN public.video_clips.duration
        IS 'duration in seconds';
    
    COMMENT ON COLUMN public.video_clips.debug_video_uri
        IS 'i.e. server://directory/subdirectory/video.mp4';
    
    COMMENT ON CONSTRAINT no_duplicate_clips ON public.video_clips
        IS 'prevent two entries of same video clip';
    
    COMMENT ON CONSTRAINT positive_clip_duration ON public.video_clips
        IS 'make sure all clip durations are positive';
""")

# create features table
cur.execute("""
    CREATE TABLE public.features
    (
        id BIGSERIAL PRIMARY KEY,
        video_clip bigint NOT NULL,
        dnn_stream character varying COLLATE pg_catalog."default" NOT NULL,
        dnn_stream_split smallint NOT NULL,
        name character varying(80) COLLATE pg_catalog."default" NOT NULL,
        dnn_weights_uri character varying(4096) COLLATE pg_catalog."default" NOT NULL,
        dnn_spec_uri character varying(4096),
        feature_vector double precision[] NOT NULL,
        CONSTRAINT no_duplicate_feature_entries UNIQUE (video_clip, dnn_stream, dnn_stream_split, name),
        CONSTRAINT dnn_stream_fkey FOREIGN KEY (dnn_stream)
            REFERENCES public.dnn_streams (type) MATCH SIMPLE
            ON UPDATE NO ACTION
            ON DELETE NO ACTION,
        CONSTRAINT video_clip_fkey FOREIGN KEY (video_clip)
            REFERENCES public.video_clips (id) MATCH SIMPLE
            ON UPDATE NO ACTION
            ON DELETE NO ACTION,
        CONSTRAINT split_gt_0 CHECK (dnn_stream_split > 0) NOT VALID
    )
    WITH (
        OIDS = FALSE
    )
    TABLESPACE pg_default;

    COMMENT ON COLUMN public.features.video_clip
        IS 'id in the video_clips table for the clip corresponding to this row';

    COMMENT ON COLUMN public.features.dnn_stream
        IS 'dnn stream name from dnn_streams table signifying which if the streams in the multi-stream DNN this row belongs';

    COMMENT ON COLUMN public.features.dnn_stream_split
        IS 'split number of dnn_stream, ensemble averaging is over splits of a given dnn stream';

    COMMENT ON COLUMN public.features.name
        IS 'should match the feature name in the DNN (e.g., caffe) model';

    COMMENT ON COLUMN public.features.dnn_weights_uri
        IS 'location of the weights file for the DNN model used to compute the feature';

    COMMENT ON COLUMN public.features.feature_vector
        IS 'feature vector';
""")
conn.commit()

# Close communication with database
cur.close()
conn.close()
