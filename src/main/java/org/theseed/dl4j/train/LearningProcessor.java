package org.theseed.dl4j.train;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.time.DurationFormatUtils;
import org.apache.commons.text.TextStringBuilder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.kohsuke.args4j.Option;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.dl4j.train.Trainer.Type;
import org.theseed.io.Shuffler;
import org.theseed.reports.IValidationReport;

public class LearningProcessor extends ModelProcessor {

    // FIELDS

    /** normalization object */
    private DataNormalization normalizer;
    /** training results */
    private RunStats results;
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(LearningProcessor.class);

    // COMMAND-LINE OPTIONS

    /** number of iterations to run for each input batch */
    @Option(name = "-n", aliases = { "--iter" }, metaVar = "1000", usage = "number of iterations per batch")
    protected int iterations;
    /** size of each input batch */
    @Option(name = "-b", aliases = { "--batchSize" }, metaVar = "1000", usage = "size of each input batch")
    protected int batchSize;
    /** maximum number of batches to read, or -1 to read them all */
    @Option(name = "-x", aliases = { "--maxBatches" }, metaVar = "6", usage = "maximum number of batches to read")
    protected int maxBatches;
    /** method to use for training */
    @Option(name = "--method", metaVar = "epoch", usage = "strategy for processing of training set")
    protected Trainer.Type method;
    /** early-stop limit */
    @Option(name = "--earlyStop", aliases = {
            "--early" }, metaVar = "100", usage = "early stop max useless iterations (0 to turn off)")
    protected int earlyStop;

    /**
     * Set the defaults and perform initialization for the parameters.
     */
    public void setDefaults() {
        this.help = false;
        this.iterations = 1000;
        this.batchSize = 500;
        this.testSize = 2000;
        this.maxBatches = Integer.MAX_VALUE;
        this.seed = (int) (System.currentTimeMillis() & 0xFFFFF);
        this.setChannelMode(false);
        this.metaCols = "";
        this.channelCount = 1;
        this.method = Type.EPOCH;
        this.earlyStop = 200;
        this.modelName = null;
        this.comment = null;
        this.idCol = null;
        // Clear the rating value and the normalizer.
        this.bestRating = 0.0;
        this.normalizer = null;
    }

    /**
     * Train and save the current model.
     *
     * @param model				model to train
     * @param runStats			RunStats object for choosing the best model
     * @param trainer			trainer for training the model
     * @param progressMonitor 	monitor for training progress
     *
     * @throws IOException
     * @throws InterruptedException
     */
    public void trainModel(MultiLayerNetwork model, RunStats runStats, Trainer trainer, ITrainReporter progressMonitor) throws IOException, InterruptedException {
        this.reader.setBatchSize(this.batchSize);
        long start = System.currentTimeMillis();
        log.info("Starting trainer.");
        trainer.trainModel(model, this.reader, getTestingSet(), runStats, progressMonitor);
        runStats.setDuration(DurationFormatUtils.formatDuration(System.currentTimeMillis() - start, "mm:ss"));
        this.results = runStats;
    }

    /**
     * If a model was created, save it to the model directory.
     *
     * @throws IOException
     */
    protected void saveModel() throws IOException {
        // Here we save the model.
        if (results.getSaveCount() > 0) {
            log.info("Saving model to {}.", this.modelName);
            ModelSerializer.writeModel(results.getBestModel(), this.modelName, true, normalizer);
        }
    }

    /**
     * @return the best model found during training
     */
    public MultiLayerNetwork getBestModel() {
        return this.results.getBestModel();
    }


    /**
     * Write the accuracy report.
     *
     * @param bestModel		model chosen for the report
     * @param buffer		text string buffer for report output
     * @param runStats		object describing run
     */
    public void accuracyReport(MultiLayerNetwork bestModel, TextStringBuilder buffer, RunStats runStats) {
        // Now we evaluate the model on the test set: compare the output to the actual
        // values.
        Evaluation eval = runStats.evaluateModel(bestModel, this.getTestingSet(), this.getLabels());
        // We need the output and the testing set labels for comparison.
        INDArray output = runStats.getOutput();
        INDArray expect = this.getTestingSet().getLabels();
        produceAccuracyReport(buffer, eval, output, expect);
        // Finally, save the accuracy in case SearchProcessor is running us.
        this.bestRating = runStats.getBestRating();
    }

    /**
     * @return a string describing the formula for a single-layer regression model with one result
     *
     * @param model		the model to dump
     */
    public String describeModel(MultiLayerNetwork model) {
        TextStringBuilder retVal = new TextStringBuilder();
        retVal.appendNewLine();
        org.deeplearning4j.nn.api.Layer layer = model.getLayers()[0];
        // Get the coefficients.
        INDArray weights = layer.paramTable().get("W");
        INDArray biases = layer.paramTable().get("b");
        // Get the corresponding column names.
        List<String> names = this.reader.getFeatureNames();
        // Remember the start of the current line.
        int line = 0;
        boolean first = true;
        for (int i = 0; i < weights.length(); i++) {
            double value = weights.getDouble(i);
            // Add this part of the formula.  Note that a negative weight is turned to
            // subtraction except on the first entry.
            if (value != 0.0) {
                if (! first) {
                    String sep = " + ";
                    if (value < 0) {
                        sep = " - ";
                        value = -value;
                    }
                    retVal.append(sep);
                }
                retVal.append("%4.2g*%s", value, names.get(i));
                first = false;
            }
            // Start the next line if we're getting long.
            if (retVal.length() - line >= 70) {
                retVal.appendNewLine().appendPadding(8, ' ');
                line = retVal.length();
            }
        }
        // Add the bias and close the formula.
        retVal.append(" + %4.2g = Confidence%n", biases.getDouble(0));
        return retVal.toString();
    }

    /**
     * @return a string describing the parameters of the specified model, layer by layer
     *
     * @param model		the model to dump
     */
    public String dumpModel(MultiLayerNetwork model) {
        TextStringBuilder retVal = new TextStringBuilder();
        retVal.appendNewLine();
        retVal.appendln("Model Parameter Summary");
        org.deeplearning4j.nn.api.Layer[] layers = model.getLayers();
        for (org.deeplearning4j.nn.api.Layer layer : layers) {
            retVal.appendln("Layer %4d: %s", layer.getIndex(), layer.type());
            Map<String, INDArray> params = layer.paramTable();
            for (String pType : params.keySet()) {
                INDArray pValue = params.get(pType);
                String shape = ArrayUtils.toString(pValue.shape());
                double min = Double.MAX_VALUE;
                double max = -Double.MAX_VALUE;
                double total = 0.0;
                long count = 0;
                long badCount = 0;
                for (long i = 0; i < pValue.length(); i++) {
                    double value = pValue.getDouble(i);
                    if (! Double.isFinite(value))
                        badCount++;
                    else {
                        if (min > value) min = value;
                        if (max < value) max = value;
                        total += value;
                        count++;
                    }
                }
                retVal.append("     %-12s: %-20s", pType, shape);
                if (count == 0)
                    retVal.appendln(" has no finite parameters");
                else {
                    retVal.append(" min = %12.4g, mean = %12.4g, max = %12.4g",
                            min, total / count, max);
                    if (badCount > 0)
                        retVal.appendln(", %d infinite values", badCount);
                    else
                        retVal.appendNewLine();
                }
            }
        }
        retVal.appendNewLine();
        return retVal.toString();
    }

    public LearningProcessor() {
        super();
    }

    /**
     * @return the recommended number of iterations
     */
    public int getIterations() {
        return iterations;
    }

    /**
     * @return the size of an example batch
     */
    public int getBatchSize() {
        return batchSize;
    }

    /**
     * @return the maximum number of batches to process
     */
    public int getMaxBatches() {
        return maxBatches;
    }

    /**
     * @return the early-stop limit
     */
    public int getEarlyStop() {
        return this.earlyStop;
    }

    /**
     * @return the testingSet
     */
    public DataSet getTestingSet() {
        return testingSet;
    }

    /**
     * @return the input normalizer
     */
    protected DataNormalization getNormalizer() {
        return normalizer;
    }

    /**
     * Store a new normalizer.
     *
     * @param normalizer the normalizer to set
     */
    protected void setNormalizer(DataNormalization normalizer) {
        this.normalizer = normalizer;
        this.reader.setNormalizer(normalizer);
    }

    /**
     * @return the trial file base name
     */
    protected String getTrialName() {
        return "trials.log";
    }

    /**
     * @return the model for this model directory
     *
     * @throws IOException
     */
    public MultiLayerNetwork readModel() throws IOException {
        if (this.modelName == null)
            this.modelName = new File(this.modelDir, "model.ser");
        log.info("Reading model from {}.", this.modelName);
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(this.modelName, false);
        DataNormalization normalizer = ModelSerializer.restoreNormalizerFromFile(this.modelName);
        this.normalizer = normalizer;
        return model;
    }

    @Override
    public void setupTraining() throws IOException {
    }

    @Override
    public void setAllDefaults() {
    }

    @Override
    protected void testModelPredictions(IValidationReport reporter, Shuffler<String> inputData) throws IOException {
    }

    @Override
    protected void initializeModelParameters() throws FileNotFoundException, IOException {
    }

    /**
     * @return TRUE if the model is valid, FALSE if it has overflowed
     *
     * @param model		the model to check
     */
    public boolean checkModel(MultiLayerNetwork model) {
        boolean retVal = true;
        org.deeplearning4j.nn.api.Layer[] layers = model.getLayers();
        for (org.deeplearning4j.nn.api.Layer layer : layers) {
            Map<String, INDArray> params = layer.paramTable();
            for (String pType : params.keySet()) {
                INDArray pValue = params.get(pType);
                long count = 0;
                for (long i = 0; i < pValue.length(); i++) {
                    double value = pValue.getDouble(i);
                    if (Double.isFinite(value))
                        count++;
                }
                if (count == 0)
                    retVal = false;
            }
        }
        return retVal;
    }

}
