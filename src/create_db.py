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
        dnn_stream character varying(80) COLLATE pg_catalog."default" NOT NULL,
        max_number_of_splits smallint,
        CONSTRAINT dnn_streams_dnn_stream_unique UNIQUE (dnn_stream)
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
    INSERT INTO dnn_streams (dnn_stream, max_number_of_splits)
    VALUES (%s, %s), (%s, %s), (%s, %s);
    """,
            ['rgb', 3, 'optical flow', 3, 'warped_optical_flow', 3]
)
conn.commit()

# create features table
cur.execute("""
    CREATE TABLE public.features
    (
        id bigint NOT NULL DEFAULT nextval('features_id_seq'::regclass),
        video_clip_id bigint NOT NULL,
        dnn_stream character varying COLLATE pg_catalog."default" NOT NULL,
        dnn_stream_split smallint NOT NULL,
        feature_name character varying(80) COLLATE pg_catalog."default" NOT NULL,
        dnn_weights_file_uri character varying(256) COLLATE pg_catalog."default" NOT NULL,
        feature double precision[] NOT NULL,
        CONSTRAINT features_pkey PRIMARY KEY (id),
        CONSTRAINT no_duplicate_feature_entries UNIQUE (video_clip_id, dnn_stream, dnn_stream_split, feature_name),
        CONSTRAINT cnn_stream_fkey FOREIGN KEY (dnn_stream)
            REFERENCES public.dnn_streams (dnn_stream) MATCH SIMPLE
            ON UPDATE NO ACTION
            ON DELETE NO ACTION,
        CONSTRAINT video_clip_id_fkey FOREIGN KEY (video_clip_id)
            REFERENCES public.video_clips (id) MATCH SIMPLE
            ON UPDATE NO ACTION
            ON DELETE NO ACTION,
        CONSTRAINT split_gt_0 CHECK (dnn_stream_split > 0) NOT VALID
    )
    WITH (
        OIDS = FALSE
    )
    TABLESPACE pg_default;
    
    COMMENT ON COLUMN public.features.video_clip_id
        IS 'id in the video_clips table for the clip corresponding to this row';
    
    COMMENT ON COLUMN public.features.dnn_stream
        IS 'cnn stream name from cnn_streams table signifying which if the streams in the multi-stream CNN this row belongs';
    
    COMMENT ON COLUMN public.features.dnn_stream_split
        IS 'split number of cnn_stream, ensemble averaging is over splits of a given cnn stream';
    
    COMMENT ON COLUMN public.features.feature_name
        IS 'should match the feature name in the CNN (e.g., caffe) model';
    
    COMMENT ON COLUMN public.features.dnn_weights_file_uri
        IS 'location of the weights file for the CNN model used to compute the feature';
    
    COMMENT ON COLUMN public.features.feature
        IS 'feature vector';
""")

# create video_clips table
cur.execute("""
    CREATE TABLE public.video_clips
    (
        id bigint NOT NULL DEFAULT nextval('video_clips_id_seq'::regclass),
        video character varying(256) COLLATE pg_catalog."default" NOT NULL,
        clip integer NOT NULL,
        clip_duration integer NOT NULL DEFAULT 10,
        video_uri character varying(256) COLLATE pg_catalog."default",
        clip_notes character varying(25600) COLLATE pg_catalog."default",
        CONSTRAINT video_clips_pkey PRIMARY KEY (id),
        CONSTRAINT no_duplicate_clips UNIQUE (video, clip, clip_duration),
        CONSTRAINT positive_clip_duration CHECK (clip_duration > 0)
    )
    WITH (
        OIDS = FALSE
    )
    TABLESPACE pg_default;
    
    ALTER TABLE public.video_clips
        OWNER to torres;
    COMMENT ON TABLE public.video_clips
        IS 'clips of videos for searching';
    
    COMMENT ON COLUMN public.video_clips.video
        IS 'name of video';
    
    COMMENT ON COLUMN public.video_clips.clip_duration
        IS 'duration in seconds';
    
    COMMENT ON COLUMN public.video_clips.video_uri
        IS 'i.e. server://directory/subdirectory/video.mp4';
    
    COMMENT ON CONSTRAINT no_duplicate_clips ON public.video_clips
        IS 'prevent two entries of same video clip';
    
    COMMENT ON CONSTRAINT positive_clip_duration ON public.video_clips
        IS 'make sure all clip durations are positive';
""")
conn.commit()
# Close communication with database
cur.close()
conn.close()