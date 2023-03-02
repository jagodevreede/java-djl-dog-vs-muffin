package org.acme;

import ai.djl.Application;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.repository.zoo.Criteria;
import ai.djl.training.util.ProgressBar;
import org.slf4j.Logger;

public class DjlResnet extends DjlAbstractLearner {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(DjlResnet.class);

    @Override
    protected Criteria.Builder<Image, Classifications> getModelBuilder() {
        Criteria.Builder<Image, Classifications> builder =
                Criteria.builder()
                        .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                        .setTypes(Image.class, Classifications.class)
                        .optProgress(new ProgressBar())
                        .optArtifactId("resnet");
        builder.optGroupId("ai.djl.mxnet");
        builder.optFilter("layers", "50");
        builder.optFilter("flavor", "v1");
        return builder;
    }

    public static void main(String[] args) throws Exception {
        log.info("Starting Resnet model");
        new DjlResnet().start(args);
    }

}
