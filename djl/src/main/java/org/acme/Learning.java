package org.acme;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.basicmodelzoo.cv.classification.ResNetV1;
import ai.djl.basicmodelzoo.cv.classification.VGG;
import ai.djl.metric.Metrics;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.SymbolBlock;
import ai.djl.nn.core.Linear;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.util.ProgressBar;

import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import static org.acme.Constants.*;

public class Learning {

    public static void main(String[] args) throws Exception {
        ImageFolder dataset = loadDataset(TRAINING_SET);
        ImageFolder validationset = loadDataset(VALIDATION_SET);

        try (Model model = getModel();
             Trainer trainer = model.newTrainer(getTrainingConfig())) {
            Path modelDir = Paths.get("model");
            trainer.setMetrics(new Metrics());

            trainer.initialize(new Shape(1, 3, IMAGE_WIDTH, IMAGE_HEIGHT));

            //EasyTrain.fit(trainer, 1, dataset, validationset);
            AtomicReference<Double> bestValidationLoss = new AtomicReference<>(99999.9);
            EarlyStoppingFit earlyStoppingFit =
                    new EarlyStoppingFit(20, 0.2, 3,
                            9 * 60, 0.1, 3);
            earlyStoppingFit.addCallback((m, epoch, validationLoss) -> {
                try {
                    if (validationLoss < bestValidationLoss.get()) {
                        model.save(modelDir, "my-demo-model-res151-e-" + epoch + "-v-" + validationLoss);
                        bestValidationLoss.set(validationLoss);
                    }
                } catch (final IOException ioe) {
                    ioe.printStackTrace();
                }
            });
            earlyStoppingFit.fit(trainer, dataset, validationset);

            model.save(modelDir, "my-demo-model-res151");
            saveLabels(modelDir, dataset.getSynset());
        }
    }

    private static void saveLabels(Path modelDir, List<String> synset) throws IOException {
        final Path labelFile = modelDir.resolve("synset.txt");
        try (Writer writer = Files.newBufferedWriter(labelFile)) {
            writer.write(String.join("\n", synset));
        }
    }

    private static TrainingConfig getTrainingConfig() {
        return new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .optOptimizer(Optimizer.adam().build())
                .addEvaluator(new Accuracy())
                .addTrainingListeners(TrainingListener.Defaults.logging(1));
    }

    private static Model getModel() {
        Model model = Model.newInstance(MODEL_NAME);

        Block resNet50 = ResNetV1.builder()
                .setImageShape(new Shape(3, IMAGE_WIDTH, IMAGE_HEIGHT))
                .setNumLayers(50)
                .setOutSize(NUM_OF_OUTPUT)
                .build();

        model.setBlock(resNet50);

//        Block vgg16 = VGG.builder()
//                .setNumLayers(16)
//                .setOutSize(NUM_OF_OUTPUT)
//                .build();
//        model.setBlock(vgg16);

        return model;
    }

    private static Model getTransferLearningModel() throws ModelNotFoundException, MalformedModelException, IOException {
        Criteria.Builder<Image, Classifications> builder = getModelBuilder();
        Model model = builder.build().loadModel();
        SequentialBlock newBlock = new SequentialBlock();
        SymbolBlock block = (SymbolBlock) model.getBlock();
        block.removeLastBlock();
        // freeze original model
        block.freezeParameters(true);
        newBlock.add(block);
        // the original model don't include the flatten so apply the flatten here
        newBlock.add(Blocks.batchFlattenBlock());
        newBlock.add(Linear.builder().setUnits(NUM_OF_OUTPUT).build());
        model.setBlock(newBlock);
        return model;
    }

    private static Criteria.Builder<Image, Classifications> getModelBuilder() {
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




    private static ImageFolder loadDataset(String folder) throws IOException {
        ImageFolder dataset = ImageFolder.builder()
                .setRepositoryPath(Paths.get(folder))
                .addTransform(new Resize(IMAGE_WIDTH, IMAGE_HEIGHT))
                .addTransform(new ToTensor())
                .setSampling(8, true)
                .build();
        dataset.prepare(new ProgressBar());
        return dataset;
    }
}
