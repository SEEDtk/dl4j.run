/**
 *
 */
package org.theseed.dl4j.train;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.text.TextStringBuilder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.theseed.utils.ICommand;

/**
 * This class reads a training set and creates and saves a multi-layered model.  It takes as
 * input a training set in a tab-delimited file, a class definition file, and a set of tuning parameters.
 * It will output a trained model and a normalizer, and will display evaluation statistics.
 *
 * It differs from ClassTrainingProcessor in that the output is a series of numbers.
 *
 * The positional parameter is the name of the model directory.  The model file (model.ser),
 * and the list of class labels (labels.txt) go in this directory.  The directory must exist,
 * and contain the following two files
 *
 * labels.txt
 * 		a text file containing the output column labels, one per line with no additional text
 *
 * training.tbl
 * 		the training set, a tab-delimited file with the feature values and the labels.  The
 * 		label column defaults to the last one, but this can be changed.  The file must have
 * 		headers.
 *
 * If the model directory also contains a "channels.tbl" file, then the input will be processed in
 * channel mode.  The aforementioned file must be tab-delimited with headers.  The first column of
 * each record should be a string, and the remaining columns should be a vector of floating-point
 * numbers.  In this case, the input is considered to be two-dimensional.  Each input string is
 * replaced by the corresponding vector.  The result can be used as a kind of one-hot representation
 * of the various strings, but it can be more complicated.  For example, if the input is DNA nucleotides,
 * an ambiguity code would contain fractional numbers in multiple positions of the vector.
 *
 * The standard output will contain evaluations and logs.  A snapshot of the input parameters and the
 * evaluation results will be appended to the file "trials.log".
 *
 * The following command-line options are supported.
 *
 * -n	number of iterations to run on each batch; the default is 1000
 * -b	size of each batch; the default is 500
 * -t	size of the testing set, which is the first batch read and is used to compute
 * 		normalization; the default is 2000
 * -w	the width of each hidden layer; this is a comma-delimited list with one number per
 * 		layer, in order; if no hidden layers are specified (use "none" for the parameter),
 * 		the input layers will feed directly into the output; a width of 0 creates an
 * 		element-wise multiplication layer
 * -u	bias updater coefficient, used to determine the starting speed of the model-- a
 * 		higher value makes the learning faster, a lower value makes it more accurate; the
 * 		default is 0.2
 * -x	maximum number of batches to process (not including the testing set); use this to
 * 		run smaller input sets for debugging the tuning parameters; the default is MAX_INTEGER,
 * 		indicating all batches should be run
 * -s	seed value to use for random number generation; the default is to use the last
 * 		20 bits of the current time
 * -l	loss function for output layer; the default is "mcxent"
 * -i	name of the training set file; the default is "training.tbl" in the model directory
 * -a	activation function for hidden layers; this should be the name of one of the
 * 		Activation class enum values; the default is "relu"
 * -r	learning rate; this should be between 1e-1 and 1e-6; the default is 1e-3
 * -z	gradient normalization strategy; the default is "none"
 *
 * --prefer		model preference-- SCORE, RSQUARED, or PEARSON
 * --bound		a minimum bound; a pseudo-accuracy will be produced based on how many in each output column
 * 				are on the correct side of the bound; the default is 0.5
 * --meta		a comma-delimited list of the metadata columns; these columns are ignored during training; the
 * 				default is none
 * --raw		if specified, the data is not normalized
 * --regMode	regularization mode-- GAUSS, LINEAR, L2, NONE; the default is GAUSS
 * --regFactor	regularization factor; a higher value makes the model less sensitive, a lower value
 * 				more sensitive; the default is 0.3
 * --cnn		if specified, then the input layer will be a convolution layer with the
 * 				specified kernel width; otherwise, it will be a standard dense layer;
 * 				specifying multiple comma-delimited values creates multiple convolution
 * 				layers
 * --lstm		if specified, then the specified number of LSTM input layers will be
 * 				put into the model; if convolution layers are specified, they will preceed the LSTM
 * 				layers
 * --init		activation type for the input layer; the default is HARDTANH
 * --comment	a comment to display at the beginning of the trial log
 * --method		training strategy to use-- epoch (small datasets) or batch (large datasets); the default
 * 				is EPOCH
 * --batch		use batch normalization before the hidden layers
 * --start		weight initialization algorithm; the default is XAVIER
 * --updater	update algorithm to use; the default is ADAM
 * --bUpdater	bias update algorithm to use; the default is NESTEROVS
 * --name		name for the model file; the default is "model.ser" in the model directory
 * --balanced	specifies balanced hidden layers with the specified number of layers; a value of 0 (the default)
 * 				turns this option off; any other value overrides "-w"; the layers are in decreasing size
 * 				order with a near-constant size reduction
 * --weights	comma-delimited list of weights for the loss function, in label order; the default is
 * 				computed according to the proportions in the testing set
 * --earlyStop	maximum number of iterations allowed with no accuracy improvement; if this
 * 				limit is exceeded, the run is terminated; a value of 0 causes the value
 * 				to be set impossibly high; the default is 200
 *
 * For a convolution input layer, the following additional parameters are used.
 *
 * --sub		indicates a subsampling layer will be used with the indicated kernel/stride size to
 * 				reduce the output from the convolution layer; the default is 1, meaning no subsampling
 * --filters	specifies the number of filters to use during convolution; each filter represents a pass
 * 				over the input with different trial parameters; the output size is multiplied by the
 * 				number of filters; a separate filter count can be specified for each convolution layer
 * 				by separating the values with commas; the last value will be repeated if the number of
 * 				filters specified is less than the number of convolution layers
 * --stride		a comma-delimited list indicating the stride for each convolution layer; if there
 * 				are fewer strides than convolution layers, the last one will be used for the rest;
 * 				the default is 1
 *
 * The following are utility options
 *
 * --parms		name of a file to contain a dump of the current parameters
 * --help		display the command-line options and parameters
 *
 * @author Bruce Parrello
 *
 */
public class RegressionTrainingProcessor extends TrainingProcessor implements ICommand {

    /** optimization preference */
    @Option(name="--prefer", metaVar="SCORE", usage="optimization preference")
    private RunStats.RegressionType prefer;
    /** accuracy testing bound */
    @Option(name="--bound", metaVar="0.9", usage="accuracy testing bound")
    private double bound;

    @Override
    protected void writeParms(File outFile) throws IOException {
        PrintWriter writer = new PrintWriter(outFile);
        String typeList = Stream.of(RunStats.RegressionType.values()).map(RunStats.RegressionType::name).collect(Collectors.joining(", "));
        writer.format("## Valid optimization preferences are %s.%n", typeList);
        writer.format("--prefer %s\t# optimization preference%n", this.prefer);
        writer.format("--bound %g\t# threshold for pseudo-accuracy%n", this.bound);
        writeModelParms(writer);
        writer.close();
    }

    @Override
    public boolean parseCommand(String[] args) {
        boolean retVal = false;
        // Set the defaults.
        this.prefer = RunStats.RegressionType.SCORE;
        this.bound = 0.5;
        this.setDefaults();
        this.setModelDefaults();
        // Parse the command line.
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);
            if (this.help) {
                parser.printUsage(System.err);
            } else {
                // Verify the model directory and read the labels.
                setupTraining(null);
                // Configure the input for regression.
                this.reader.setRegressionColumns();
                // Read in the testing set.
                readTestingSet();
                // Set up the common parameters.
                initializeModelParameters();
                // We made it this far, we can run the application.
                retVal = true;
            }
        } catch (CmdLineException e) {
            System.err.println(e.getMessage());
            // For parameter errors, we display the command usage.
            parser.printUsage(System.err);
        } catch (IOException e) {
            System.err.println(e.getMessage());
        }
        return retVal;
    }

    @Override
    public void run() {
        try {
            // Create the model.
            MultiLayerNetwork model = buildModel();
            // Train the model.
            Trainer trainer = Trainer.create(this.method, this, log);
            RunStats runStats = RunStats.createR(model, prefer, trainer, this);
            this.trainModel(model, runStats, trainer);
            this.saveModel();
            // Display the configuration.
            MultiLayerNetwork bestModel = runStats.getBestModel();
            TextStringBuilder parms = displayModel(runStats);
            if (runStats.getSaveCount() == 0) {
                parms.appendNewLine();
                parms.appendln("MODEL FAILED DUE TO OVERFLOW OR UNDERFLOW.");
                this.clearRating();
            } else {
                this.regressionReport(bestModel, parms, runStats);
            }
            // Add the summary.
            parms.appendln(bestModel.summary(getInputShape()));
            // Add the parameter dump.
            parms.append(this.dumpModel(bestModel));
            // Output the result.
            String report = parms.toString();
            log.info(report);
            RunStats.writeTrialReport(this.getTrialFile(), this.comment, report);
        } catch (Exception e) {
            e.printStackTrace(System.err);
        }
    }

    /**
     * Write the performance report.  This is different from classification because of the way accuracy is measured.
     *
     * @param model		the model to be reported
     * @param buffer	the TextStringBuilder to contain the report
     * @param runStats	the RunStats containing the model results
     */
    private void regressionReport(MultiLayerNetwork model, TextStringBuilder buffer, RunStats runStats) {
        // Evaluate the model.
        RegressionEvaluation eval = runStats.scoreModel(model, getTestingSet(), getLabels());
        // Write the header.
        buffer.appendln("%n%-21s %11s %11s %11s %11s %11s %11s", "label", "MAE", "MSE", "PC", "R Squared", "Pseudo-Acc", "Bound-Acc");
        buffer.appendln(StringUtils.repeat('-', 93));
        // We need the output and the testing set labels for the pseudo-accuracy.
        INDArray output = runStats.getOutput();
        INDArray expect = this.getTestingSet().getLabels();
        // Write the stats for each column.
        for (int i = 0; i < this.getLabels().size(); i++) {
            String label = this.getLabels().get(i);
            int goodCount = 0;
            int closeCount = 0;
            for (long r = 0; r < this.testSize; r++) {
                double expected = expect.getDouble(r, i);
                double actual = output.getDouble(r, i);
                boolean eDiff = (expected >= this.bound);
                boolean oDiff = (actual >= this.bound);
                if (eDiff == oDiff) goodCount++;
                if (Math.round(actual) == expected) closeCount++;
            }
            String accuracy = LearningProcessor.formatRatio(goodCount, this.testSize);
            String closeness = LearningProcessor.formatRatio(closeCount, this.testSize);
            buffer.appendln("%-21s %11.4f %11.4f %11.4f %11.4f %11s %11s", label, eval.meanAbsoluteError(i), eval.meanSquaredError(i),
                    eval.pearsonCorrelation(i), eval.rSquared(i), accuracy, closeness);
        }
        if (this.getLabels().size() > 1) {
            // Write the average stats.
            buffer.appendln("%-21s %11.4f %11.4f %11.4f %11.4f", "AVERAGE", eval.averageMeanAbsoluteError(),
                    eval.averageMeanSquaredError(), eval.averagePearsonCorrelation(), eval.averageRSquared());
        } else if (model.getnLayers() == 1 && this.getChannelCount() == 1) {
            // Here we have a single-layer, single-label model.  Write the formula.
            buffer.appendln(this.describeModel(model));
        }
        // Save the score in case this is a search.
        this.setRating(runStats.getBestRating());
    }

    @Override
    protected Activation getOutActivation() {
        return this.getLossFunction().getOutActivation(Activation.IDENTITY);
    }

    /**
     * @return the threshold for pseudo-accuracy
     */
    public double getBound() {
        return bound;
    }

}
