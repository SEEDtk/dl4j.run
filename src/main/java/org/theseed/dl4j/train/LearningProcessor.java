package org.theseed.dl4j.train;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
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
    private double bestRating;
    /** normalization object */
    private DataNormalization normalizer;
    /** training results */
    private RunStats results;


    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(ClassTrainingProcessor.class);

    // COMMAND LINE PARAMETER
    /** help option */
    @Option(name = "-h", aliases = { "--help" }, help = true)
    protected boolean help;
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
     * Format a ratio for display in the evaluation metrics matrix.
     *
     * @param value		numerator
     * @param total		denominator
     *
     * @return a formatted fraction, or an empty string if the denominator is 0
     */
    protected static String formatRatio(double value, int total) {
        String retVal = "";
        if (total > 0) {
            retVal = String.format("%11.4f", value / total);
        }
        return retVal;
    }
    /**
     * Set the defaults and perform initialization for the parameters.
     */
    protected void setDefaults() {
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
        // Clear the rating value and the normalizer.
        this.bestRating = 0.0;
        this.normalizer = null;
    }

    /**
     * Train and save the current model.
     *
     * @param model			model to train
     * @param runStats		RunStats object for choosing the best model
     * @param trainer		trainer for training the model
     *
     * @throws IOException
     */
    public void trainModel(MultiLayerNetwork model, RunStats runStats, Trainer trainer) throws IOException {
        this.reader.setBatchSize(this.batchSize);
        long start = System.currentTimeMillis();
        log.info("Starting trainer.");
        trainer.trainModel(model, this.reader, getTestingSet(), runStats);
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
        // Output the evaluation.
        buffer.appendln(eval.stats());
        ConfusionMatrix<Integer> matrix = eval.getConfusion();
        // We need the output and the testing set labels for comparison.
        INDArray output = runStats.getOutput();
        INDArray expect = this.getTestingSet().getLabels();
        // Analyze the negative results.
        int actualNegative = matrix.getActualTotal(0);
        if (actualNegative == 0) {
            buffer.appendln("No \"%s\" results were found.", this.getLabels().get(0));
        } else {
            double specificity = ((double) matrix.getCount(0, 0)) / actualNegative;
            buffer.appendln("Model specificity is %11.4f.%n", specificity);
        }
        // Write the header.
        buffer.appendln("%-11s %11s %11s %11s %11s %11s", "class", "accuracy", "sensitivity", "precision", "fallout", "MAE");
        buffer.appendln(StringUtils.repeat('-', 71));
        // The classification accuracy is 1 - (false negative + false positive) / total,
        // sensitivity is true positive / actual positive, precision is true positive / predicted positive,
        // and fall-out is false positive / actual negative.  The L2 Error is the trickiest.  It is the absolute
        // difference between the expected values and the output, divided by the number of examples.
        for (int i = 1; i < this.getLabels().size(); i++) {
            String label = this.getLabels().get(i);
            double accuracy = 1 - ((double) (matrix.getCount(0, i) + matrix.getCount(i,  0))) / this.testSize;
            double l1_error = 0.0;
            for (long r = 0; r < this.testSize; r++) {
                double diff = expect.getDouble(r, i) - output.getDouble(r, i);
                l1_error += Math.abs(diff);
            }
            String sensitivity = formatRatio(matrix.getCount(i, i), matrix.getActualTotal(i));
            String precision = formatRatio(matrix.getCount(i, i), matrix.getPredictedTotal(i));
            String fallout = formatRatio(matrix.getCount(0,  i), actualNegative);
            String l1Error = formatRatio(l1_error, this.testSize);
            buffer.appendln("%-11s %11.4f %11s %11s %11s %11s", label, accuracy, sensitivity, precision, fallout, l1Error);
        }
        // Finally, save the accuracy in case SearchProcessor is running us.
        this.bestRating = runStats.getBestRating();
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
     * @return the rating of the best epoch
     */
    public double getRating() {
        return bestRating;
    }

    /**
     * Store a new best rating.
     */
    public void setRating(double newValue) {
        this.bestRating = newValue;
    }

    /**
     * Reset the rating indicator to show failure to achieve a result.
     */
    public void clearRating() {
        this.bestRating = Double.NEGATIVE_INFINITY;
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
     * Initialize the reader for reading a training set.
     *
     * @oaram labelCol	column specifier for the label column, or NULL if there is none
     * @param metaList	list of metadata columns
     *
     * @throws IOException
     */
    public void initializeReader(String labelCol, List<String> metaList) throws IOException {
        // Determine the input type and get the appropriate reader.
        File channelFile = new File(this.modelDir, "channels.tbl");
        this.channelMode = channelFile.exists();
        if (! this.channelMode) {
            log.info("Normal input.");
            // Normal situation.  Read scalar values.
            this.reader = new TabbedDataSetReader(this.trainingFile, labelCol, this.getLabels(), metaList);
        } else {
            // Here we have channel input.
            Map<String, double[]> channelMap = ChannelDataSetReader.readChannelFile(channelFile);
            ChannelDataSetReader myReader = new ChannelDataSetReader(this.trainingFile, labelCol,
                    this.getLabels(), metaList, channelMap);
            this.channelCount = myReader.getChannels();
            this.reader = myReader;
            log.info("Channel input with {} channels.", this.getChannelCount());
        }
    }

    /**
     * Read in the testing set.
     */
    protected void readTestingSet() {
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
     * @oaram labelCol	column specifier for the label column, or NULL if there is none
     *
     * @throws FileNotFoundException
     * @throws IOException
     */
    public void setupTraining(String labelCol) throws FileNotFoundException, IOException {
        if (! this.modelDir.isDirectory()) {
            throw new FileNotFoundException("Model directory " + this.modelDir + " not found or invalid.");
        } else {
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
            initializeReader(labelCol, metaList);
        }
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

}
