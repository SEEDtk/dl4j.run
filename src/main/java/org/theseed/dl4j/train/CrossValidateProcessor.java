/**
 *
 */
package org.theseed.dl4j.train;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.stream.Collectors;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.apache.commons.text.TextStringBuilder;
import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.basic.ICommand;
import org.theseed.basic.ParseFailureException;
import org.theseed.counters.RegressionStatistics;
import org.theseed.counters.Shuffler;
import org.theseed.dl4j.TabbedDataSetReader;
import org.theseed.io.LineReader;
import org.theseed.io.TabbedLineReader;
import org.theseed.reports.NullTrainReporter;
import org.theseed.reports.TestValidationReport;
import org.theseed.utils.Parms;

/**
 * This class performs cross-validation on a training set.  The set is divided into equal portions (folds)
 * and a model is trained once for each fold, with the fold used as the testing set and the remaining data used for
 * training.  Each model is then evaluated against the entire training/testing set and the best model is saved.
 *
 * The rating of the model is a prediction error, computed according to the model type.
 *
 * The positional parameter is the name of the model directory.
 *
 * The command-line options are as follows.
 *
 * -h	display command-line usage
 * -k	number of folds to use (default 10)
 * -t	type of model (CLASS or REGRESSION, default CLASS)
 *
 * --parms 	name of the parameter file (default is "parms.prm" in the model directory)
 *
 * @author Bruce Parrello
 *
 */
public class CrossValidateProcessor implements ICommand {

    // FIELDS
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(CrossValidateProcessor.class);
    /** list of strings in training/testing set */
    private Shuffler<String> mainFile;
    /** processor being validated */
    private ITrainingProcessor trainingProcessor;
    /** error tracker */
    RegressionStatistics errorTracker;
    /** testing set size */
    private int testSize;
    /** array index of best model */
    private int bestIdx;
    /** error rating of best model */
    private double bestError;
    /** parameter list object */
    private Parms parms;
    /** array of fold errors */
    private double[] foldErrors;
    /** array of fold statistics */
    private double[][] foldStats;
    /** progress monitor */
    private ITrainReporter progressMonitor;

    // COMMAND-LINE OPTIONS

    /** help option */
    @Option(name="-h", aliases={"--help"}, help=true)
    private boolean help;

    /** type of model */
    @Option(name="--type", aliases={"-t"}, usage="type of model")
    private ModelType modelType;

    /** number of folds to test */
    @Option(name = "-k", aliases = { "--folds" }, metaVar = "5", usage = "Number of folds to use for cross-validation")
    private int foldK;

    /** parameter file */
    @Option(name = "--parms", metaVar="parms.prm", usage="parameter file (if not the default)")
    private File parmFile;

    /** model directory */
    @Argument(index=0, metaVar="modelDir", usage="model directory", required=true)
    private File modelDir;

    /**
     * Create a new, blank cross-validation processor.
     */
    public CrossValidateProcessor() {
        this.progressMonitor = new NullTrainReporter();
    }

    /**
     * Create a cross-validation processor with a specific progress monitor.
     *
     * @param monitor	progress monitor to use for reporting
     */
    public CrossValidateProcessor(ITrainReporter monitor) {
        this.progressMonitor = monitor;
    }

    @Override
    public boolean parseCommand(String[] args) {
        boolean retVal = false;
        // Parse the command line.
        CmdLineParser parser = new CmdLineParser(this);
        try {
            this.help = false;
            this.foldK = 10;
            this.modelType = ModelType.CLASS;
            this.parmFile = null;
            parser.parseArgument(args);
            if (this.help) {
                parser.printUsage(System.err);
            } else {
                // Verify that the fold size is reasonable.
                if (this.foldK < 2)
                    throw new ParseFailureException("Invalid k-fold " + Integer.toString(this.foldK) + ".  Must be 2 or greater.");
                // Verify the model directory.
                if (! this.modelDir.isDirectory())
                    throw new FileNotFoundException("Model directory " + this.modelDir + " not found or invalid.");
                // Read the parameter file.
                if (this.parmFile == null)
                    this.parmFile = new File(this.modelDir, "parms.prm");
                log.info("Reading parameters from {}.", this.parmFile);
                // Read the parameter file.
                if (! this.parmFile.canRead())
                    throw new FileNotFoundException("Parameter file " + this.parmFile + " not found or unreadable.");
                else {
                    this.parms = new Parms(this.parmFile);
                    // Extract the training file.
                    String trainingName = this.parms.getValue("--training");
                    File trainingFile;
                    if (trainingName.isEmpty())
                        trainingFile = new File(this.modelDir, "training.tbl");
                    else
                        trainingFile = new File(trainingName);
                    // Read it into memory.
                    log.info("Reading all data lines from {}.", trainingFile);
                    this.mainFile = new Shuffler<String>(1000);
                    try (LineReader mainReader = new LineReader(trainingFile)) {
                        this.mainFile.addSequence(mainReader);
                    }
                    this.mainFile.trimToSize();
                    log.info("{} data lines found in file.", this.mainFile.size() - 1);
                    // Compute the testing set size.
                    this.testSize = (this.mainFile.size() - 1) / this.foldK;
                    if (this.testSize < 1) this.testSize = 1;
                    log.info("Testing size is {}.", this.testSize);
                    this.parms.set("--testSize", this.testSize);
                    // Here we are good.
                    retVal = true;
                }
            }
        } catch (CmdLineException | ParseFailureException e) {
            System.err.println(e.toString());
            // For parameter errors, we display the command usage.
            parser.printUsage(System.err);
            this.progressMonitor.showResults("Parameter error: " + e.toString());
        } catch (IOException e) {
            System.err.println(e.toString());
        }
        return retVal;
    }

    @Override
    public void run() {
        try {
            // Denote we have no best result yet.
            this.bestIdx = 1;
            this.bestError = Double.MAX_VALUE;
            // Set up the error tracking.  Note that we have an extra slot in the array because we start at position 1.
            this.foldErrors = new double[this.foldK + 1];
            this.foldStats = new double[this.foldK + 1][];
            this.errorTracker = new RegressionStatistics(this.foldK);
            // Create the training processor.
            this.trainingProcessor = ModelType.create(this.modelType);
            this.trainingProcessor.setModelDir(this.modelDir);
            // Set the progress monitor.
            this.trainingProcessor.setProgressMonitor(this.progressMonitor);
            try {
                RunLog.writeTrialMarker(this.trainingProcessor.getTrialFile(), "Cross-Validate");
            } catch (IOException e) {
                log.error("Error writing trial file: {}", e.toString());
            }
            // Set the defaults.
            this.trainingProcessor.setupParameters(this.parms, this.modelDir);
            // This will hold the validation report statistic titles.
            String[] statTitles = null;
            // Prevent the model from saving.
            this.trainingProcessor.setSearchMode();
            for (int k = 1; k <= this.foldK; k++) {
                // Fix the comment.
                String comment = String.format("Cross-validation fold %d.", k);
                this.trainingProcessor.setComment(comment);
                // Read the data and run the training.
                TabbedDataSetReader myReader = this.trainingProcessor.openReader(this.mainFile);
                this.trainingProcessor.configureTraining(myReader);
                this.progressMonitor.showMessage(comment);
                this.trainingProcessor.run();
                // Compute the accuracy.  Note we reread the input.
                TestValidationReport testErrorReport = this.trainingProcessor.getTestReporter();
                String idCol = this.trainingProcessor.getIdCol();
                if (idCol != null) {
                    try (TabbedLineReader reader = new TabbedLineReader(this.mainFile)) {
                        Collection<String> trained = this.trainingProcessor.getTrainingMeta(reader);
                        testErrorReport.setupIdCol(this.modelDir, idCol, this.trainingProcessor.getMetaList(), trained);
                    }
                }
                IPredictError errors = this.trainingProcessor.testBestPredictions(this.mainFile, testErrorReport);
                double thisError = testErrorReport.getError();
                log.info("Mean error for this model was {}.", thisError);
                if (thisError < this.bestError) {
                    this.bestIdx = k;
                    this.bestError = thisError;
                    log.info("Model is the best so far.");
                    this.trainingProcessor.saveModelForced();
                    if (this.trainingProcessor.getIdCol() != null)
                        this.trainingProcessor.saveTrainingMeta(this.mainFile);
                    this.progressMonitor.showResults(this.trainingProcessor.getResultReport());
                } else {
                    log.info("Best so far is fold {} with error {}.", this.bestIdx, this.bestError);
                }
                // Now we need to track this model's performance.  First, insure we have titles.
                if (statTitles == null)
                    statTitles = errors.getTitles();
                // Store the mean error.
                this.errorTracker.add(thisError);
                this.foldErrors[k] = thisError;
                // Store the auxiliary stats.
                this.foldStats[k] = errors.getStats();
                // Shuffle the input for the next fold.
                this.mainFile.rotate(1, -this.testSize);
            }
            log.info("Best result was fold {} with error {}.", this.bestIdx, this.bestError);
            // Finish the regression statistics.
            this.errorTracker.finish();
            // Build the cross-validation report.
            String boundary = StringUtils.repeat('=', 25);
            TextStringBuilder buffer = new TextStringBuilder(25 * (this.foldK + 6) + 50);
            buffer.appendNewLine();
            buffer.appendln(boundary);
            buffer.appendln(String.format("%4s  %14s ", "Fold", "Error")
                    + Arrays.stream(statTitles).map(x -> String.format(" %14s", x)).collect(Collectors.joining()));
            buffer.appendNewLine();
            for (int k = 1; k <= this.foldK; k++) {
                char flag = (k == this.bestIdx ? '*' : ' ');
                buffer.appendln(String.format("%4d. %14.8f%c", k, this.foldErrors[k], flag)
                        + Arrays.stream(this.foldStats[k]).mapToObj(x -> String.format(" %14.8f", x)).collect(Collectors.joining()));
            }
            buffer.appendln(boundary);
            buffer.appendNewLine();
            buffer.appendln(String.format("Cross-validation metrics: trimean %14.8f, trimmed mean %14.8f, IQR %14.8f", this.errorTracker.trimean(),
                    this.errorTracker.trimmedMean(0.2), this.errorTracker.iqr()));
            // Write the report and log it.
            String report = buffer.toString();
            RunLog.writeTrialReport(this.trainingProcessor.getTrialFile(), "Summary of Cross-Validation", report);
            log.info(report);
        } catch (Exception e) {
            e.printStackTrace(System.err);
            this.progressMonitor.showResults(ExceptionUtils.getStackTrace(e));
        }
    }

    /**
     * @return the error information for the cross-validation
     */
    public RegressionStatistics getResults() {
        return this.errorTracker;
    }

}
