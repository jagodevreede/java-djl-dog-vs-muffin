package org.acme;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Stream;

import static org.acme.Constants.*;

public class ModelVerification {
    private static final Logger logger = LoggerFactory.getLogger(ModelVerification.class);

    private final Map<String, Map<String, Integer>> matrix = new HashMap<>();

    public ModelVerification(Model model) {
        // define a translator for pre and post processing
        // out of the box this translator converts images to ResNet friendly ResNet 18 shape
        Translator<Image, Classifications> translator = ImageClassificationTranslator.builder()
                .addTransform(new Resize(IMAGE_WIDTH, IMAGE_HEIGHT))
                .addTransform(new ToTensor())
                .optApplySoftmax(true)
                .build();
        // run the inference using a Predictor
        try (Predictor<Image, Classifications> predictor = model.newPredictor(translator)) {
            Stream.of(new File(VALIDATION_SET).listFiles(filter -> filter.isDirectory())).forEach(folder -> {
                Stream.of(folder.listFiles(filter -> filter.isFile())).forEach(file -> {
                    predictFile(predictor, folder, file);
                });
            });
        }
        printConfusionMatrix();
    }

    private void predictFile(Predictor<Image, Classifications> predictor, File folder, File file) {
        try {
            Path imageFile = file.toPath();
            Image img = ImageFactory.getInstance().fromFile(imageFile);
            // holds the probability score per label
            Classifications predictResult = predictor.predict(img);
            Classifications.Classification classification = predictResult.topK(1).get(0);
            String className = classification.getClassName();
            var outcome = matrix.getOrDefault(folder.getName(), new HashMap<>());
            var count = outcome.getOrDefault(className, 0);
            outcome.put(className, count + 1);
            matrix.put(folder.getName(), outcome);
            if (!folder.getName().equals(className)) {
                logger.info("{} should be {} but was {} with probability {}", file, folder.getName(), className, classification.getProbability());
            }
        } catch (TranslateException | IOException e) {
            logger.error(e.getMessage(), e);
        }
    }

    private void printConfusionMatrix() {
        // Find the length of the longest key in the outer map and the inner maps
        int maxKeyLength = matrix.keySet().stream().mapToInt(String::length).max().orElse(0) +1;

        // Calculate the diagonal sum and the total sum
        int diagonalSum = 0;
        int totalSum = 0;
        for (String actual : matrix.keySet()) {
            Map<String, Integer> innerMap = matrix.get(actual);
            for (String predicted : matrix.keySet()) {
                int count = innerMap.getOrDefault(predicted, 0);
                totalSum += count;
                if (actual.equals(predicted)) {
                    diagonalSum += count;
                }
            }
        }

        // Print header row
        System.out.printf("%-" + maxKeyLength + "s", "");
        for (String actual : matrix.keySet()) {
            System.out.printf("%-" + (maxKeyLength + 1) + "s", actual);
        }
        System.out.printf("%-13s%n", "Accuracy");

        // Print rows
        for (String actual : matrix.keySet()) {
            System.out.printf("%-" + maxKeyLength + "s", actual);
            Map<String, Integer> innerMap = matrix.get(actual);
            int rowSum = 0;
            int correctCount = innerMap.getOrDefault(actual, 0);
            for (String predicted : matrix.keySet()) {
                int count = innerMap.getOrDefault(predicted, 0);
                rowSum += count;
                System.out.printf("%-" + (maxKeyLength + 1) + "d", count);
            }
            double accuracy = (double) correctCount / rowSum;
            System.out.printf("%-13.2f%n", accuracy * 100);
        }

        double accuracy = (double) diagonalSum / totalSum;
        System.out.printf("Total accuracy: %-13.2f%n", accuracy * 100);
    }
}
