package org.acme;

import java.nio.file.Path;
import java.nio.file.Paths;

public class Constants {

    public static final String TRAINING_SET = "./training_set";
    public static final String VALIDATION_SET = "./test-set";
    public static final int IMAGE_WIDTH = 224;
    public static final int IMAGE_HEIGHT = 224;
    public static final long NUM_OF_OUTPUT = 2;

    public static final String MODEL_NAME = "dog-vs-muffin";

    public static final Path modelDir = Paths.get("models");
}
