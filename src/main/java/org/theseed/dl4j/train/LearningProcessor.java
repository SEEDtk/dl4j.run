package org.theseed.dl4j.train;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.time.DurationFormatUtils;
import org.apache.commons.text.TextStringBuilder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.Option;
import org.nd4j.evaluation.classification.ConfusionMatrix;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.dl4j.ChannelDataSetReader;
import org.theseed.dl4j.TabbedDataSetReader;
import org.theseed.dl4j.train.Trainer.Type;

public class LearningProcessor {

    // FIELDS

    /** input file reader */
    protected TabbedDataSetReader reader;
    /** array of labels */
    private List<String> labels;
    /** testing set */
    private DataSet testingSet;
    /** number of input channels */
    private int channelCount;
    /** TRUE if we have channel input */
    private boolean channelMode;
    /** best accuracy */
    private double bestAccuracy;
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(TrainingProcessor.class);

    // COMMAND LINE PARAMETER
    /** help option */
    @Option(name = "-h", aliases = { "--help" }, help = true)
    protected boolean help;
    /** name or index of the label column */
    @Option(name = "-c", aliases = { "--col" }, metaVar = "0", usage = "input column containing class")
    protected String labelCol;
    /** number of iterations to run for each input batch */
    @Option(name = "-n", aliases = { "--iter" }, metaVar = "1000", usage = "number of iterations per batch")
    protected int iterations;
    /** size of each input batch */
    @Option(name = "-b", aliases = { "--batchSize" }, metaVar = "1000", usage = "size of each input batch")
    protected int batchSize;
    /** size of testing set */
    @Option(name = "-t", aliases = { "--testSize" }, metaVar = "1000", usage = "size of the testing set")
    protected int testSize;
    /** maximum number of batches to read, or -1 to read them all */
    @Option(name = "-x", aliases = { "--maxBatches" }, metaVar = "6", usage = "maximum number of batches to read")
    protected int maxBatches;
    /** initialization seed */
    @Option(name = "-s", aliases = { "--seed" }, metaVar = "12765", usage = "random number seed")
    protected int seed;
    /** comma-delimited list of metadata column names */
    @Option(name = "--meta", metaVar = "name,date", usage = "comma-delimited list of metadata columns")
    protected String metaCols;
    /** comment to display in trial log */
    @Option(name = "--comment", metaVar = "changed bias rate", usage = "comment to display in trial log")
    protected String comment;
    /** method to use for training */
    @Option(name = "--method", metaVar = "epoch", usage = "strategy for processing of training set")
    protected Trainer.Type method;
    /** early-stop limit */
    @Option(name = "--earlyStop", aliases = {
            "--early" }, metaVar = "100", usage = "early stop max useless iterations (0 to turn off)")
    protected int earlyStop;
    /** model file name */
    @Option(name = "--name", usage = "model file name (default is model.ser in model directory)")
    protected File modelName;
    /** indicates category 0 is negative */
    @Option(name = "--other", usage = "show metrics assuming category 0 is a negative result")
    protected boolean otherMode;
    /** optimization preference */
    @Option(name = "--prefer", metaVar = "SCORE", usage = "model aspect to optimize during search")
    protected RunStats.OptimizationType preference;
    /** input training set */
    @Option(name="-i", aliases={"--input"}, metaVar="training.tbl",
            usage="input training set file")
    protected File trainingFile;
    /** model directory */
    @Argument(index = 0, metaVar = "modelDir", usage = "model directory", required = true)
    protected File modelDir;

    /**
     * Format a ratio for display in the evaluation metrics matrix.
     *
     * @param count		numerator
     * @param total		denominator
     *
     * @return a formatted fraction, or an empty string if the denominator is 0
     */
    protected static String formatRatio(int count, int total) {
        String retVal = "";
        if (total > 0) {
            retVal = String.format("%11.4f", ((double) count) / total);
        }
        return retVal;
    }

    /**
     * Set the defaults and perform initialization for the parameters.
     */
    protected void setDefaults() {
        this.help = false;
        this.labelCol = "1";
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
        this.otherMode = false;
        this.preference = RunStats.OptimizationType.ACCURACY;
        // Clear the accuracy value.
        this.bestAccuracy = 0.0;
    }

    /**
     * Train and save the current model.
     *
     * @param normalizer	normalizer to use for the data
     * @param model			model to train
     *
     * @return a RunStats describing the training results
     *
     * @throws IOException
     */
    public RunStats trainModel(DataNormalization normalizer, MultiLayerNetwork model) throws IOException {
        this.reader.setBatchSize(this.batchSize);
        long start = System.currentTimeMillis();
        Trainer myTrainer = Trainer.create(this.method, this, log);
        log.info("Starting trainer.");
        RunStats runStats = myTrainer.trainModel(model, this.reader, getTestingSet());
        runStats.setDuration(DurationFormatUtils.formatDuration(System.currentTimeMillis() - start, "mm:ss"));
        // Here we save the model.
        if (runStats.getSaveCount() > 0) {
            log.info("Saving model to {}.", this.modelName);
            ModelSerializer.writeModel(runStats.getBestModel(), this.modelName, true, normalizer);
        }
        return runStats;
    }

    /**
     * Write the accuracy report.
     *
     * @param bestModel		model chosen for the report
     * @param buffer		text string buffer for report output
     */
    public void accuracyReport(MultiLayerNetwork bestModel, TextStringBuilder buffer) {
        // Now we evaluate the model on the test set: compare the output to the actual
        // values.
        Evaluation eval = Trainer.evaluateModel(bestModel, this.getTestingSet(), this.getLabels());
        // Output the evaluation.
        buffer.append(eval.stats());
        ConfusionMatrix<Integer> matrix = eval.getConfusion();
        // This last thing is the table of scores for each prediction.  This only makes sense if we have
        // an "other" mode.
        if (this.otherMode) {
            int actualNegative = matrix.getActualTotal(0);
            if (actualNegative == 0) {
                buffer.appendln("No \"%s\" results were found.", this.getLabels().get(0));
            } else {
                double specificity = ((double) matrix.getCount(0, 0)) / actualNegative;
                buffer.appendln("Model specificity is %11.4f.%n", specificity);
            }
            buffer.appendln("%-11s %11s %11s %11s %11s", "class", "accuracy", "sensitivity", "precision", "fallout");
            buffer.appendln(StringUtils.repeat('-', 59));
            // The classification accuracy is 1 - (false negative + false positive) / total,
            // sensitivity is true positive / actual positive, precision is true positive / predicted positive,
            // and fall-out is false positive / actual negative.
            for (int i = 1; i < this.getLabels().size(); i++) {
                String label = this.getLabels().get(i);
                double accuracy = 1 - ((double) (matrix.getCount(0, i) + matrix.getCount(i,  0))) / this.testSize;
                String sensitivity = formatRatio(matrix.getCount(i, i), matrix.getActualTotal(i));
                String precision = formatRatio(matrix.getCount(i, i), matrix.getPredictedTotal(i));
                String fallout = formatRatio(matrix.getCount(0,  i), actualNegative);
                buffer.appendln("%-11s %11.4f %11s %11s %11s", label, accuracy, sensitivity, precision, fallout);
            }
        }
        // Finally, save the accuracy in case SearchProcessor is running us.
        this.bestAccuracy = eval.accuracy();
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
                else
                    retVal.appendln(" min = %12.4g, mean = %12.4g, max = %12.4g, %d infinite values",
                            min, total / count, max, badCount);
            }
        }
        retVal.appendNewLine();
        return retVal.toString();
    }

    public LearningProcessor() {
        super();
    }

    /**
     * @return the optimization preference
     */
    public RunStats.OptimizationType getPreference() {
        return preference;
    }

    /**
     * @return the accuracy of the best epoch
     */
    public double getAccuracy() {
        return bestAccuracy;
    }

    /**
     * Reset the accuracy indicator to show failure to achieve a result.
     */
    public void clearAccuracy() {
        this.bestAccuracy = 0;
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
     * @return the labels
     */
    public List<String> getLabels() {
        return labels;
    }

    /**
     * @param labels the labels to set
     */
    public void setLabels(List<String> labels) {
        this.labels = labels;
    }

    /**
     * @return the channelMode
     */
    public boolean isChannelMode() {
        return channelMode;
    }

    /**
     * @param channelMode the channelMode to set
     */
    public void setChannelMode(boolean channelMode) {
        this.channelMode = channelMode;
    }

    /**
     * @param metaList
     * @throws IOException
     */
    public void initializeReader(List<String> metaList) throws IOException {
        // Determine the input type and get the appropriate reader.
        File channelFile = new File(this.modelDir, "channels.tbl");
        this.channelMode = channelFile.exists();
        if (! this.channelMode) {
            log.info("Normal input.");
            // Normal situation.  Read scalar values.
            this.reader = new TabbedDataSetReader(this.trainingFile, this.labelCol, this.getLabels(), metaList);
        } else {
            // Here we have channel input.
            HashMap<String, double[]> channelMap = ChannelDataSetReader.readChannelFile(channelFile);
            ChannelDataSetReader myReader = new ChannelDataSetReader(this.trainingFile, this.labelCol,
                    this.getLabels(), metaList, channelMap);
            this.channelCount = myReader.getChannels();
            this.reader = myReader;
            log.info("Channel input with {} channels.", this.getChannelCount());
        }
        // Get the testing set.
        log.info("Reading testing set (size = {}).", this.testSize);
        this.reader.setBatchSize(this.testSize);
        this.testingSet = this.reader.next();
        if (! this.reader.hasNext()) {
            log.warn("Training set contains only test data. Batch size = {} but {} records in input.",
                    this.testSize, this.getTestingSet().numExamples());
            throw new IllegalArgumentException("No training data.");
        }
    }

    /**
     * Set up the training file for processing.
     *
     * @throws FileNotFoundException
     * @throws IOException
     */
    public void setupTraining() throws FileNotFoundException, IOException {
        // Read in the labels from the label file.
        File labelFile = new File(this.modelDir, "labels.txt");
        if (! labelFile.exists())
            throw new FileNotFoundException("Label file not found in " + this.modelDir + ".");
        this.setLabels(TabbedDataSetReader.readLabels(labelFile));
        log.info("{} labels read from label file.", this.getLabels().size());
        // Parse the metadata column list.
        List<String> metaList = Arrays.asList(StringUtils.split(this.metaCols, ','));
        // Finally, we initialize the input to get the label and metadata columns handled.
        if (this.trainingFile == null) {
            this.trainingFile = new File(this.modelDir, "training.tbl");
            if (! this.trainingFile.exists())
                throw new FileNotFoundException("Training file " + this.trainingFile + " not found.");
        }
        initializeReader(metaList);
    }

    /**
     * @return the testingSet
     */
    public DataSet getTestingSet() {
        return testingSet;
    }

    /**
     * @return the channelCount
     */
    public int getChannelCount() {
        return channelCount;
    }

}
