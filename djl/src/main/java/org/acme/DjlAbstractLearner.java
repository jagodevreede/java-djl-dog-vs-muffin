package org.acme;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.metric.Metrics;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.SymbolBlock;
import ai.djl.nn.core.Linear;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.training.*;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

abstract class DjlAbstractLearner {
    private final Logger log = org.slf4j.LoggerFactory.getLogger(this.getClass());

    private static final String FLOWER_DIR = "./training_set";

    private static final int IMAGE_WIDTH = 224;
    private static final int IMAGE_HEIGHT = 224;
    private static int batchSize = 16;
    private final String MODEL_NAME = "dog-vs-muffin-" + this.getClass().getSimpleName();
    private static final long NUM_OF_OUTPUT = 2;
    private static final int EPOCHS = 10;

    private static final Path modelDir = Paths.get("models");

    void start(String... args) throws Exception {
        if (args.length == 1) {
            batchSize = Integer.parseInt(args[0]);
        }
        log.info("Starting training with batch size: {}", batchSize);
        ImageFolder dataset = initDataset(FLOWER_DIR);
        // Split the dataset set into training dataset and validate dataset
        RandomAccessDataset[] datasets = dataset.randomSplit(8, 2);

        // set loss function, which seeks to minimize errors
        // loss function evaluates model's predictions against the correct answer (during training)
        // higher numbers are bad - means model performed poorly; indicates more errors; want to
        // minimize errors (loss)
        Loss loss = Loss.softmaxCrossEntropyLoss();

        // setting training parameters (ie hyperparameters)
        TrainingConfig config = setupTrainingConfig(loss);

        try (Model model = getModel(); // empty model instance to hold patterns
             Trainer trainer = model.newTrainer(config)) {
            log.info("Model setup complete");
            // metrics collect and report key performance indicators, like accuracy
            trainer.setMetrics(new Metrics());

            Shape inputShape = new Shape(1, 3, IMAGE_HEIGHT, IMAGE_HEIGHT);

            // initialize trainer with proper input shape
            trainer.initialize(inputShape);

            final long startTime = System.currentTimeMillis();
            // find the patterns in data
            EasyTrain.fit(trainer, EPOCHS, datasets[0], datasets[1]);

            log.info("Model build complete in " + (System.currentTimeMillis() - startTime) / 1000 + "sec");

            // set model properties
            TrainingResult result = trainer.getTrainingResult();
            model.setProperty("Epoch", String.valueOf(EPOCHS));
            model.setProperty("Accuracy", String.format("%.5f", result.getValidateEvaluation("Accuracy")));
            model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));

            // save the model after done training for inference later
            model.save(modelDir, MODEL_NAME);

            // save labels into model directory
            saveLabels(modelDir, dataset.getSynset());
            log.info("Total build complete in " + (System.currentTimeMillis() - startTime) / 1000 + "sec");
        }
    }

    public void saveLabels(Path modelDir, List<String> synset) throws IOException {
        final Path labelFile = modelDir.resolve(MODEL_NAME + ".txt");
        try (Writer writer = Files.newBufferedWriter(labelFile)) {
            writer.write(String.join("\n", synset));
        }
    }

    protected abstract Criteria.Builder<Image, Classifications> getModelBuilder();

    private Model getModel() throws IOException, ModelNotFoundException, MalformedModelException {
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
        //model.getBlock().clear();
        return model;
    }

    private TrainingConfig setupTrainingConfig(Loss loss) {
        return new DefaultTrainingConfig(loss)
                .addEvaluator(new Accuracy())
                .optExecutorService()
                .addTrainingListeners(TrainingListener.Defaults.logging(1));
    }

    private ImageFolder initDataset(String datasetRoot) throws IOException {
        ImageFolder dataset =
                ImageFolder.builder()
                        // retrieve the data
                        .setRepositoryPath(Paths.get(datasetRoot))
                        .optMaxDepth(2)
                        .addTransform(new Resize(IMAGE_WIDTH, IMAGE_HEIGHT))
                        .addTransform(new ToTensor())
                        // random sampling; don't process the data in order
                        .setSampling(batchSize, true)
                        .build();

        dataset.prepare(new ProgressBar());
        return dataset;
    }

}
