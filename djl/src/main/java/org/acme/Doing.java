package org.acme;

import ai.djl.Model;
import ai.djl.basicmodelzoo.cv.classification.ResNetV1;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.translate.Translator;

import java.nio.file.Path;
import java.nio.file.Paths;

import static org.acme.Constants.*;

public class Doing {
    public static void main(String[] args) throws Exception {
        try (Model model = getModel()) {
            model.load(modelDir, MODEL_NAME);

            Translator<Image, Classifications> translator = createTranslator();

            try (Predictor<Image, Classifications> predictor = model.newPredictor(translator)) {
                Path imageFile = Paths.get(VALIDATION_SET + "/Muffin/1.png");
                Image img = ImageFactory.getInstance().fromFile(imageFile);
                Classifications classifications = predictor.predict(img);
                System.out.println(classifications.topK());
            }
        }
    }

    private static Translator<Image, Classifications> createTranslator() {
        return ImageClassificationTranslator.builder()
                .addTransform(new Resize(IMAGE_WIDTH, IMAGE_HEIGHT))
                .addTransform(new ToTensor())
                .optApplySoftmax(true)
                .build();
    }

    private static Model getModel() {
        Model model = Model.newInstance(MODEL_NAME);

        Block resNet50 = ResNetV1.builder()
                .setImageShape(new Shape(3, IMAGE_WIDTH, IMAGE_HEIGHT))
                .setNumLayers(50)
                .setOutSize(NUM_OF_OUTPUT)
                .build();

        model.setBlock(resNet50);
        return model;
    }
}
