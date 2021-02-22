/**
 *
 */
package org.theseed.dl4j.train;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.time.DurationFormatUtils;
import org.apache.commons.text.TextStringBuilder;
import org.kohsuke.args4j.Option;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.dl4j.DistributedOutputStream;
import org.theseed.dl4j.Impact;
import org.theseed.dl4j.TabbedDataSetReader;
import org.theseed.dl4j.decision.RandomForest;
import org.theseed.io.Shuffler;
import org.theseed.reports.ClassTestValidationReport;
import org.theseed.reports.ClassValidationReport;
import org.theseed.reports.IValidationReport;
import org.theseed.reports.TestValidationReport;

/**
 * This is the base class for anything that trains a model.  Unlike LearningProcessor, the model does
 * not have to be a neural net, nor does it require an iterative search.  In addition, the hyper-parameters
 * are dramatically different.
 *
 * The positional parameter is the name of the model directory.  The directory should contain the training data
 * in the file "training.tbl" and the label values in the file "labels.txt".  The model itself will be stored
 * in "model.ser".  The following command-line parameters are supported.
 *
 * -c	index (1-based) or name of the column containing the classification labels; the
 * 		default is 1 (first column)
 * -t	size of the testing set, which is the first batch read and is used to compute
 * 		normalization; the default is 2000
 * -s	seed value to use for random number generation; the default is to use the last
 * 		20 bits of the current time
 * -i	name of the training set file; the default is "training.tbl" in the model directory
 *
 * --meta			a comma-delimited list of the metadata columns; these columns are ignored during training; the
 * 					default is none
 * --comment		a comment to display at the beginning of the trial log
 * --name			name for the model file; the default is "model.ser" in the model directory
 * --parms			name of a file to contain a dump of the current parameters
 * --help			display the command-line options and parameters
 * --id				name of the ID column in the training file; this is also a meta-data column and must be unique
 * --method			randomization method (BALANCED, UNIQUE, RANDOM)
 * --maxFeatures	number of features to use at each tree node
 * --nEstimators	number of trees in the forest
 * --minSplit		minimum number of examples allowed in a choice node
 * --maxDepth		maximum tree depth
 * --sampleSize		number of examples to use for each tree's subset
 *
 * @author Bruce Parrello
 *
 */
public class RandomForestTrainProcessor extends ModelProcessor implements ITrainingProcessor {

    // FIELDS
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(RandomForestTrainProcessor.class);
    /** trained model */
    private RandomForest model;
    /** hyper-parameters */
    private RandomForest.Parms hParms;
    /** result report */
    private String resultReport;

    // COMMAND-LINE OPTIONS

    /** parameter dump */
    @Option(name="--parms", usage="write command-line parameters to a configuration file")
    private File parmFile;

    /** label column ID or number */
    @Option(name = "-c", aliases = { "--col" }, metaVar = "result", usage = "label column name")
    private String labelCol;

    /** randomization method for choosing example rows */
    @Option(name = "--method", usage = "randomization method")
    private RandomForest.Method method;

    /** number of features to use at each tree node */
    @Option(name = "--maxFeatures", metaVar = "10", usage = "number of features to interrogate at each splitting tree node")
    private int maxFeatures;

    /** minimum number of examples allowed in a choice node */
    @Option(name = "--minSplit", metaVar = "2", usage = "minimum number of examples allowed in a splitting tree node")
    private int minSplit;

    /** number of trees in the forest */
    @Option(name = "--nEstimators", metaVar = "100", usage = "number of trees in the forest")
    private int nEstimators;

    /** maximum permissible tree depth */
    @Option(name = "--maxDepth", metaVar = "10", usage = "maximum tree depth")
    private int maxDepth;

    /** number of samples to use for each tree's subset */
    @Option(name = "--sampleSize", metaVar = "100", usage = "sample size to use for each tree")
    private int sampleSize;

    @Override
    public boolean parseCommand(String[] args) {
        boolean retVal = false;
        // Set the defaults.
        this.setAllDefaults();
        // Parse the command line.
        try {
            if (this.parseArgs(args)) {
                this.setupTraining();
                // Verify the model directory and read the labels.
                TabbedDataSetReader myReader = this.openReader(this.trainingFile, this.labelCol);
                // Configure the model for training.
                this.configureTraining(myReader);
                // We made it this far, we can run the application.
                retVal = true;
            }
        } catch (IOException e) {
            System.err.println(e.getMessage());
        }
        return retVal;
    }

    @Override
    public void run() {
        try {
            // Read the testing set.
            this.readTestingSet();
            // Convert it to decision tree format.
            RandomForestTrainProcessor.flattenDataSet(this.testingSet);
            // Now we read the training set.
            DataSet trainingSet = this.readTrainingSet();
            // Build the model from the training set.
            long start = System.currentTimeMillis();
            log.info("Building the model.");
            this.model = new RandomForest(trainingSet, this.hParms);
            // Test the accuracy.
            log.info("Testing the model.");
            // Create a label array for output.
            INDArray predictions = this.model.predict(this.testingSet.getFeatures());
            String duration = DurationFormatUtils.formatDuration(System.currentTimeMillis() - start, "mm:ss");
            // Get the actual labels.
            INDArray expectations = this.testingSet.getLabels();
            Evaluation accuracy = new Evaluation(this.getLabels());
            accuracy.eval(expectations, predictions);
            log.info("Writing the report.");
            TextStringBuilder reportBuilder = new TextStringBuilder(800);
            // Describe the model.
            reportBuilder.appendNewLine();
            reportBuilder.appendln(
                            "Model file is %s%n" +
                            "=========================== Parameters ===========================%n" +
                            "     maxFeatures = %12d, minSplit      = %12d%n" +
                            "     nEstimators = %12d, maxDepth      = %12d%n" +
                            "     sampleSize  = %12d%n" +
                            "     --------------------------------------------------------%n" +
                            "     Randomization strategy is %s with seed %d.%n" +
                            "     %s minutes to train model.",
                           this.modelName.getCanonicalPath(),
                           this.maxFeatures, this.minSplit, this.nEstimators, this.maxDepth,
                           this.sampleSize, this.method.toString(), this.seed, duration);
            this.produceAccuracyReport(reportBuilder, accuracy, predictions, expectations);
            this.bestRating = accuracy.accuracy();
            reportBuilder.appendNewLine();
            INDArray impact = this.model.computeImpact();
            SortedSet<Impact> cols = this.computeImpactList(impact);
            // Output the 10 most impactful columns.
            reportBuilder.appendln("Impact       Column Name");
            reportBuilder.appendln("---------------------------------------------------------");
            Iterator<Impact> iter = cols.iterator();
            int count = 0;
            while (iter.hasNext() && count < 20) {
                Impact item = iter.next();
                reportBuilder.appendln("%12.4f %s", item.getImpact(), item.getName());
                count++;
            }
            reportBuilder.appendNewLine();
            String report = reportBuilder.toString();
            log.info(report);
            RunStats.writeTrialReport(this.getTrialFile(), this.comment, report);
            this.resultReport = this.comment + report;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }

    }

    /**
     * @return a sorted list of the impact values for all the columns
     *
     * @param impact	impact vector, indexed by input column
     */
    private SortedSet<Impact> computeImpactList(INDArray impact) {
        // The tricky part here is we need to compute the column names.  We go through the feature
        // columns one by one.  If we are in channel mode, each channel element gets a number suffix.
        List<String> featureNames = this.reader.getFeatureNames();
        int channels = this.getChannelCount();
        List<String> colNames = new ArrayList<String>(featureNames.size() * channels);
        for (String col : this.reader.getFeatureNames()) {
            if (! this.isChannelMode())
                colNames.add(col);
            else
                IntStream.rangeClosed(1, channels).mapToObj(i -> String.format("%s|%02d", col, i)).forEach(x -> colNames.add(x));
        }
        // Now we fill in the output set.
        SortedSet<Impact> retVal = new TreeSet<Impact>();
        for (int i = 0; i < impact.columns(); i++)
            retVal.add(new Impact(colNames.get(i), impact.getDouble(i)));
        return retVal;
    }

    /**
     * @return the full training set for this model
     */
    private DataSet readTrainingSet() {
        log.info("Reading training set.");
        List<DataSet> batches = new ArrayList<DataSet>();
        for (DataSet batch : this.reader) {
            flattenDataSet(batch);
            batches.add(batch);
        }
        DataSet retVal = DataSet.merge(batches);
        log.info("{} records read from training set.", retVal.numExamples());
        return retVal;
    }

    @Override
    public DistributedOutputStream getDistributor() {
        return new DistributedOutputStream.Discrete();
    }

    @Override
    public IValidationReport getValidationReporter(OutputStream outStream) {
        return new ClassValidationReport(outStream);
    }

    @Override
    public List<String> getLabelCols() {
        return Collections.singletonList(this.labelCol);
    }

    @Override
    public void setAllDefaults() {
        RandomForest.Parms hyperParms = new RandomForest.Parms();
        this.copyHyperParmsToOptions(hyperParms);
        this.help = false;
        this.testSize = 2000;
        this.seed = (int) (System.currentTimeMillis() & 0xFFFFF);
        this.setChannelMode(false);
        this.metaCols = "";
        this.channelCount = 1;
        this.modelName = null;
        this.comment = null;
        this.idCol = null;
        // Clear the rating value.
        this.bestRating = 0.0;
    }

    @Override
    public List<String> computeAvailableHeaders(List<String> headers, Collection<String> labels) {
        // Every header is available for a classification model.  One will be selected as the label column.
        return headers;
    }

    @Override
    public void setMetaCols(String[] cols) {
        if (cols.length > 1)
            this.metaCols = StringUtils.join(cols, ',', 0, cols.length - 1);
        else
            this.metaCols = "";
        if (cols.length > 0)
            this.labelCol = cols[cols.length - 1];
        else
            this.labelCol = "1";
    }

    @Override
   public void writeParms(File outFile) throws IOException {
       PrintWriter writer = new PrintWriter(outFile);
       writer.format("--col %s\t# input column for class name%n", this.labelCol);
       writeBaseModelParms(writer);
       String typeList = Stream.of(RandomForest.Method.values()).map(RandomForest.Method::name).collect(Collectors.joining(", "));
       writer.format("# Valid randomization methods are %s.%n", typeList);
       writer.format("--method %s\t# randomization method for picking subsamples%n", this.method);
       writer.format("--maxFeatures %d\t# number of features to check at each tree node%n", this.maxFeatures);
       writer.format("--minSplit %d\t# minimum number of examples allowed in a splitting tree node%n", this.minSplit);
       writer.format("--nEstimators %d\t# number of trees in the forest%n", this.nEstimators);
       writer.format("--maxDepth %d\t# maximum allowable tree depth%n", this.maxDepth);
       writer.format("--sampleSize %d\t# number of examples to use in each subsample%n", this.sampleSize);
       writer.close();
   }

    @Override
    public TabbedDataSetReader openReader(List<String> strings) throws IOException {
        return this.openReader(strings, this.labelCol);
    }

    @Override
    public TabbedDataSetReader openReader(File inFile) throws IOException {
        return this.openReader(inFile, this.labelCol);
    }

    @Override
    public void configureTraining(TabbedDataSetReader myReader) throws IOException {
        this.initializeReader(myReader);
        this.initializeTraining();
    }

    @Override
    public TestValidationReport getTestReporter() {
        return new ClassTestValidationReport();
    }

    @Override
    public IPredictError testBestPredictions(List<String> mainFile, IValidationReport testErrorReport)
            throws IOException {
        // Get access to the input data.
        TabbedDataSetReader batches = this.openReader(mainFile, this.labelCol);
        // Initialize the error predictor.
        IPredictError retVal = new ClassPredictError(this.getLabels());
        // Initialize the output report.
        testErrorReport.startReport(this.getMetaList(), this.getLabels());
        // Loop through the data, making predictions.
        for (DataSet batch : batches) {
            INDArray features = flattenFeatures(batch.getFeatures());
            INDArray output = this.model.predict(features);
            retVal.accumulate(batch.getLabels(), output);
            testErrorReport.reportOutput(batch.getExampleMetaData(String.class), batch.getLabels(), output);
        }
        // Finish predicting.
        retVal.finish();
        testErrorReport.finishReport(retVal);
        // Close the report output stream.
        testErrorReport.close();
        // Return the evaluation of the predictions.
        return retVal;
    }

    @Override
    public String getResultReport() {
        return this.resultReport;
    }

    @Override
    public void saveModelForced() throws IOException {
        this.model.save(this.modelName);
    }

    @Override
    public void setupTraining() throws IOException {
        this.setupTraining(this.labelCol);
    }

    @Override
    protected void testModelPredictions(IValidationReport reporter, Shuffler<String> inputData) throws IOException {
        this.model = RandomForest.load(this.modelName);
        this.testBestPredictions(inputData, reporter);
    }

    @Override
    protected void initializeModelParameters() throws FileNotFoundException, IOException {
        // Copy the parameters to the hyper-parameters.
        this.hParms = new RandomForest.Parms();
        this.hParms.setMethod(this.method);
        this.hParms.setLeafLimit(this.minSplit - 1);
        this.hParms.setMaxDepth(this.maxDepth);
        this.hParms.setnExamples(this.sampleSize);
        this.hParms.setNumFeatures(this.maxFeatures);
        this.hParms.setNumTrees(this.nEstimators);
        // Set the randomizer seed.
        RandomForest.setSeed(seed);
        // Initialize the low-level computed parameters.
        initializeBaseModelParms();
        // If the user asked for a configuration file, write it here.
        if (this.parmFile != null) writeParms(this.parmFile);
    }

    /**
     * Convert a feature array from the 4-dimensional shape used by neural nets to a standard 2 dimensions.
     *
     * @param features		feature array to reshape
     *
     * @return the incoming list of examples, flattened to a row/column matrix
     */
    public static INDArray flattenFeatures(INDArray features) {
        return features.reshape(features.size(0), features.size(1) * features.size(3));
    }

    /**
     * Convert a dataset's feature array from the 4-dimensional shape used by neural nets to a standard 2 dimensions.
     *
     * @param dataset		dataset to convert
     */
    public static void flattenDataSet(DataSet dataset) {
        dataset.setFeatures(flattenFeatures(dataset.getFeatures()));
    }

    @Override
    public void setSizeParms(int inputSize, int featureCols) {
        this.testSize = inputSize / 10;
        if (testSize < 1) testSize = 1;
        int nExamples = inputSize - testSize;
        RandomForest.Parms hyperParms = new RandomForest.Parms(nExamples, featureCols);
        copyHyperParmsToOptions(hyperParms);
    }

    /**
     * Copy hyper-parameters to this object's command-line options.
     *
     * @param hyperParms	hyper-parameters to copy
     */
    protected void copyHyperParmsToOptions(RandomForest.Parms hyperParms) {
        this.method = hyperParms.getMethod();
        this.maxDepth = hyperParms.getMaxDepth();
        this.nEstimators = hyperParms.getNumTrees();
        this.minSplit = hyperParms.getLeafLimit() + 1;
        this.maxFeatures = hyperParms.getNumFeatures();
        this.sampleSize = hyperParms.getNumExamples();
    }

}
