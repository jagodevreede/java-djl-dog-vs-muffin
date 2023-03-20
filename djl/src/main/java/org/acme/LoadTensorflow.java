package org.acme;

import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Translator;

import java.nio.file.Path;
import java.nio.file.Paths;

public class LoadTensorflow {
    public static void main(String[] args) throws Exception {
        Engine.getEngine("TensorFlow");

        Criteria<Image, Classifications> criteria = Criteria.builder()
                .setTypes(Image.class, Classifications.class)
                .optModelPath(Paths.get("../ml-banchmark/python"))
                .optModelName("model/1")
                .optEngine("TensorFlow")
                .optProgress(new ProgressBar())
                .build();

        try (ZooModel<Image, Classifications> model = criteria.loadModel()) {
            Translator<Image, Classifications> translator = ImageClassificationTranslator.builder()
                    .addTransform(new Resize(150, 150))
                    .addTransform(new ToTensor())
                    .optApplySoftmax(true)
                    .build();
            // run the inference using a Predictor
            try (Predictor<Image, Classifications> predictor = model.newPredictor(translator)) {
                Path imageFile = Paths.get("/Users/jagodevreede/dl4j-examples-data/dl4j-examples/flower_photos/daisy/99306615_739eb94b9e_m.jpg");
                Image img = ImageFactory.getInstance().fromFile(imageFile);
                Classifications classifications = predictor.predict(img);
                System.out.println(classifications.topK());
            }
        }
    }

}
