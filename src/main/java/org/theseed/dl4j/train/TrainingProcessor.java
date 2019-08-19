/**
 *
 */
package org.theseed.dl4j.train;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.time.DurationFormatUtils;
import org.nd4j.evaluation.classification.ConfusionMatrix;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.misc.ElementWiseMultiplicationLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.dl4j.ChannelDataSetReader;
import org.theseed.dl4j.Regularization;
import org.theseed.dl4j.TabbedDataSetReader;
import org.theseed.dl4j.train.Trainer.Type;
import org.theseed.utils.ICommand;
import org.theseed.utils.IntegerList;


/**
 * This class reads a training set and creates and saves a multi-layered model.  It takes as
 * input a training set in a tab-delimited file, a class definition file, and a set of tuning parameters.
 * It will output a trained model and a normalizer, and will display evaluation statistics.
 *
 * The positional parameter is the name of the model directory.  The model file (model.ser),
 * and the list of class labels (labels.txt) go in this directory.  The directory must exist,
 * and contain the following two files
 *
 * labels.txt
 * 		a text file containing the classification labels, one per line with no additional text
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
 * -c	index (1-based) or name of the column containing the classification labels; the
 * 		default is 1 (first column)
 * -n	number of iterations to run on each batch; the default is 1000
 * -b	size of each batch; the default is 500
 * -t	size of the testing set, which is the first batch read and is used to compute
 * 		normalization; the default is 2000
 * -w	the width of each hidden layer; this is a comma-delimited list with one number per
 * 		layer, in order; if no hidden layers are specified, a default hidden layer whose
 * 		width is the mean of the input and output widths will be created; a width of 0 creates
 * 		an element-wise multiplication layer
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
 * --init		activation type for the input layer; the default is HARDTANH
 * --comment	a comment to display at the beginning of the trial log
 * --method		training strategy to use-- epoch (small datasets) or batch (large datasets); the default
 * 				is EPOCH
 * --batch		use batch normalization before the hidden layers
 * --weights	weight initialization algorithm; the default is XAVIER
 * --updater	update algorithm to use; the default is ADAM
 * --bUpdater	bias update algorithm to use; the default is SGD
 * --name		name for the model file; the default is "model.ser" in the model directory
 * --other		if specified, it is assumed that category 0 is the negative category; specifying this
 * 				option triggers additional output metrics
 * --balanced	specifies balanced hidden layers with the specified number of layers; a value of 0 (the default)
 * 				turns this option off; any other value overrides "-w"; the layers are in decreasing size
 * 				order with a near-constant size reduction
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
 * For training method EPOCH, the following options apply:
 *
 * --earlyStop	maximum number of iterations allowed with no accuracy improvement; if this
 * 				limit is exceeded, the run is terminated; the default of 0 causes the value
 * 				to be set equal to the iteration limit, effectively negating the parameter
 *
 * The following are utility options
 *
 * --parms		name of a file to contain a dump of the current parameters
 * --help		display the command-line options and parameters
 *
 * @author Bruce Parrello
 *
 */
public class TrainingProcessor implements ICommand {

    // FIELDS
    /** input file reader */
    private TabbedDataSetReader reader;
    /** array of labels */
    private List<String> labels;
    /** testing set */
    private DataSet testingSet;
    /** number of input channels */
    private int channelCount;
    /** TRUE if we have channel input */
    private boolean channelMode;
    /** list of convolution widths */
    private IntegerList convolutions;
    /** list of hidden layer widths */
    private IntegerList denseLayers;
    /** list of filter counts */
    private IntegerList filterSizes;
    /** list of strides */
    private IntegerList strides;
    /** actual updater learning rate */
    private double realLearningRate;
    /** best accuracy */
    private double bestAccuracy;
    /** regularization control object */
    private Regularization regulizer;

    /** logging facility */
    public static Logger log = LoggerFactory.getLogger(TrainingProcessor.class);

    // COMMAND LINE

    /** help option */
    @Option(name="-h", aliases={"--help"}, help=true)
    private boolean help;

    /** parameter dump */
    @Option(name="--parms", usage="write command-line parameters to a configuration file")
    private File parmFile;

    /** name or index of the label column */
    @Option(name="-c", aliases={"--col"}, metaVar="0",
            usage="input column containing class")
    private String labelCol;

    /** number of iterations to run for each input batch */
    @Option(name="-n", aliases={"--iter"}, metaVar="1000",
            usage="number of iterations per batch")
    private int iterations;

    /** size of each input batch */
    @Option(name="-b", aliases={"--batchSize"}, metaVar="1000",
            usage="size of each input batch")
    private int batchSize;

    /** size of testing set */
    @Option(name="-t", aliases={"--testSize"}, metaVar="1000",
            usage="size of the testing set")
    private int testSize;

    /** number of nodes in each middle layer */
    @Option(name="-w", aliases={"--widths"}, metaVar="7",
            usage="width of each hidden layer (0 for element-wise multiplication")
    private void setLayers(String layerWidths) {
        this.denseLayers = new IntegerList(layerWidths);
    }

    /** \regularization factor to prevent overfitting */
    @Option(name="--regFactor", metaVar="0.5", usage="regularization coefficient/factor")
    private double regFactor;

    /** regularization mode */
    @Option(name="--regMode", usage="regularization mode")
    private Regularization.Mode regMode;

    /** maximum number of batches to read, or -1 to read them all */
    @Option(name="-x", aliases={"--maxBatches"}, metaVar="6",
            usage="maximum number of batches to read")
    private int maxBatches;

    /** bias updater coefficient */
    @Option(name="-u", aliases={"--updateRate"}, metaVar="0.1",
            usage="bias updater coefficient")
    private double biasRate;

    /** initialization seed */
    @Option(name="-s", aliases={"--seed"}, metaVar="12765", usage="random number seed")
    private int seed;

    /** loss function */
    @Option(name="-l", aliases={"--lossFun", "--loss"}, metaVar="mse",
            usage="loss function for scoring output layer")
    private LossFunctions.LossFunction lossFunction;

    /** normalization flag */
    @Option(name="--raw", usage="suppress dataset normalization")
    private boolean rawMode;

    /** default activation function */
    @Option(name="-a", aliases= {"--activation"}, usage="activation function for hidden layers")
    private Activation activationType;

    /** initial activation function */
    @Option(name="--init", usage="activation function for input layer")
    private Activation initActivationType;

    /** input training set */
    @Option(name="-i", aliases={"--input"}, metaVar="training.tbl",
            usage="input training set file")
    private File trainingFile;

    /** learning rate */
    @Option(name="-r", aliases={"--learnRate"}, metaVar="0.1",
            usage="learning rate")
    private double learnRate;

    /** gradient normalization strategy */
    @Option(name="-z", aliases={"--gradNorm"}, metaVar="RenormalizeL2PerLayer",
            usage="gradient normalization strategy")
    private GradientNormalization gradNorm;

    /** use batch normalization */
    @Option(name="--batch", usage="use batch normalization before hidden layers")
    private boolean batchNormFlag;

    /** convolution mode */
    @Option(name="--cnn", metaVar="3", usage="convolution mode, specifying kernel size of each layer")
    private void setConvolution(String cnn) {
        this.convolutions = new IntegerList(cnn);
    }

    /** subsampling layer */
    @Option(name="--sub", metaVar="2", usage="kernel/stride of subsampling layer (if any)")
    private int subFactor;

    /** convolution output */
    @Option(name="--filters", metaVar="3", usage="number of trial filters to use for each convolution")
    private void setFilters(String filters) {
        this.filterSizes = new IntegerList(filters);
    }

    /** convolution strides */
    @Option(name="--strides", metaVar="3,2", usage="stride to use for each convolution")
    private void setStrides(String strides) {
        this.strides = new IntegerList(strides);
    }

    /** comma-delimited list of metadata column names */
    @Option(name="--meta", metaVar="name,date", usage="comma-delimited list of metadata columns")
    private String metaCols;

    /** comment to display in trial log */
    @Option(name="--comment", metaVar="changed bias rate", usage="comment to display in trial log")
    private String comment;

    /** method to use for training */
    @Option(name="--method", metaVar="epoch", usage="strategy for processing of training set")
    private Trainer.Type method;

    /** weight initialization algorithm */
    @Option(name="--weight", usage="weight initialization strategy")
    private WeightInit weightInitMethod;

    /** early-stop limit */
    @Option(name="--earlyStop", aliases={"--early"}, metaVar="100",
            usage="early stop max useless iterations (0 to turn off)")
    private int earlyStop;

    /** bias updater algorithm */
    @Option(name="--bUpdater", usage="bias gradient updater algorithm")
    private GradientUpdater.Type biasUpdateMethod;

    /** model file name */
    @Option(name="--name", usage="model file name (default is model.ser in model directory)")
    private File modelName;

    /** weight updater algorithm */
    @Option(name="--updater", usage="weight gradient updater algorithm")
    private GradientUpdater.Type weightUpdateMethod;

    /** indicates category 0 is negative */
    @Option(name="--other", usage="show metrics assuming category 0 is a negative result")
    private boolean otherMode;

    /** use balanced layers */
    @Option(name="--balanced", metaVar="4", usage="compute balanced layer widths for the specified number of layers")
    private int balancedLayers;

    /** model directory */
    @Argument(index=0, metaVar="modelDir", usage="model directory", required=true)
    private File modelDir;


    /**
     * Class to describe a fitting run.
     */
    public static class RunStats {
        private boolean errorStop;
        private int bounceCount;
        private int eventCount;
        private MultiLayerNetwork bestModel;
        private int saveCount;
        private int bestEvent;

        public RunStats(MultiLayerNetwork model) {
            this.errorStop = false;
            this.bounceCount = 0;
            this.eventCount = 0;
            this.bestModel = model;
            this.bestEvent = 0;
            this.saveCount = 0;
        }

        /** Record a score bounce. */
        public void bounce() {
            this.bounceCount++;
        }

        /** Record a batch. */
        public void event() {
            this.eventCount++;
        }

        /** Record an error. */
        public void error() {
            this.errorStop = true;
        }

        /**
         * @return the error-stop flag
         */
        public boolean isErrorStop() {
            return errorStop;
        }

        /**
         * @return the number of score bounces
         */
        public int getBounceCount() {
            return bounceCount;
        }

        /**
         * @return the number of events processed
         */
        public int getEventCount() {
            return eventCount;
        }

        /**
         * @return the best model found
         */
        public MultiLayerNetwork getBestModel() {
            return bestModel;
        }

        /**
         * @return the number of model saves
         */
        public int getSaveCount() {
            return saveCount;
        }

        /**
         * @return the event corresponding to the best model
         */
        public int getBestEvent() {
            return bestEvent;
        }

        /**
         * Store the new best model.
         * @param bestModel 	the new best model
         */
        public void setBestModel(MultiLayerNetwork bestModel) {
            this.bestModel = bestModel;
            this.bestEvent = this.eventCount;
            this.saveCount++;
        }

    }

    /**
     * Parse command-line options to specify the parameters of this object.
     *
     * @param args	an array of the command-line parameters and options
     *
     * @return TRUE if successful, FALSE if the parameters are invalid
     */
    public boolean parseCommand(String[] args) {
        boolean retVal = false;
        // Set the defaults.
        this.help = false;
        this.labelCol = "1";
        this.iterations = 1000;
        this.batchSize = 500;
        this.testSize = 2000;
        this.denseLayers = new IntegerList();
        this.regFactor = 0.3;
        this.biasRate = 0.2;
        this.learnRate = 1e-3;
        this.maxBatches = Integer.MAX_VALUE;
        this.lossFunction = LossFunctions.LossFunction.MCXENT;
        this.seed = (int) (System.currentTimeMillis() & 0xFFFFF);
        this.trainingFile = null;
        this.regMode = Regularization.Mode.GAUSS;
        this.activationType = Activation.RELU;
        this.initActivationType = Activation.HARDTANH;
        this.gradNorm = GradientNormalization.None;
        this.convolutions = new IntegerList();
        this.channelMode = false;
        this.subFactor = 1;
        this.filterSizes = new IntegerList("1");
        this.strides = new IntegerList("1");
        this.metaCols = "";
        this.channelCount = 1;
        this.parmFile = null;
        this.method = Type.EPOCH;
        this.batchNormFlag = false;
        this.earlyStop = 0;
        this.weightInitMethod = WeightInit.XAVIER;
        this.biasUpdateMethod = GradientUpdater.Type.SGD;
        this.weightUpdateMethod = GradientUpdater.Type.ADAM;
        this.modelName = null;
        this.comment = null;
        this.otherMode = false;
        this.balancedLayers = 0;
        // Clear the accuracy value.
        this.bestAccuracy = 0.0;
        // Parse the command line.
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);
            if (this.help) {
                parser.printUsage(System.err);
            } else {
                // Verify the model directory and read the labels.
                if (! this.modelDir.isDirectory()) {
                    throw new FileNotFoundException("Model directory " + this.modelDir + " not found or invalid.");
                } else {
                    // Read in the labels from the label file.
                    File labelFile = new File(this.modelDir, "labels.txt");
                    if (! labelFile.exists()) {
                        throw new FileNotFoundException("Label file not found in " + this.modelDir + ".");
                    } else {
                        this.labels = TabbedDataSetReader.readLabels(labelFile);
                        log.info("{} labels read from label file.", this.labels.size());
                        // Parse the metadata column list.
                        List<String> metaList = Arrays.asList(StringUtils.split(this.metaCols, ','));
                        // Finally, we initialize the input to get the label and metadata columns handled.
                        if (this.trainingFile == null) {
                            this.trainingFile = new File(this.modelDir, "training.tbl");
                            if (! this.trainingFile.exists())
                                throw new FileNotFoundException("Training file " + this.trainingFile + " not found.");
                        }
                        // Determine the input type and get the appropriate reader.
                        File channelFile = new File(this.modelDir, "channels.tbl");
                        this.channelMode = channelFile.exists();
                        if (! this.channelMode) {
                            log.info("Normal input.");
                            // Normal situation.  Read scalar values.
                            this.reader = new TabbedDataSetReader(this.trainingFile, this.labelCol, this.labels, metaList);
                        } else {
                            // Here we have channel input.
                            HashMap<String, double[]> channelMap = ChannelDataSetReader.readChannelFile(channelFile);
                            ChannelDataSetReader myReader = new ChannelDataSetReader(this.trainingFile, this.labelCol,
                                    this.labels, metaList, channelMap);
                            this.channelCount = myReader.getChannels();
                            this.reader = myReader;
                            log.info("Channel input with {} channels.", this.channelCount);
                        }
                        // TODO balanced layers
                        // Insure the number of filters or strides is not greater than the number of convolutions.
                        if (! this.convolutions.isEmpty()) {
                            if (this.filterSizes.size() > this.convolutions.size())
                                throw new IllegalArgumentException("Cannot have more filters than convolution layers.");
                            if (this.strides.size() > this.convolutions.size())
                                throw new IllegalArgumentException("Cannot have more strides than convolution layers.");
                        }
                        // Verify that the subfactor is in range.
                        int subChannels = this.reader.getWidth();
                        int strideFactor = this.strides.first();
                        if (! this.convolutions.isEmpty() && this.subFactor > 1) {
                            // Compute the width after the last convolution.
                            for (int cWidth : this.convolutions) {
                                subChannels = (subChannels - cWidth) / strideFactor + 1;
                                strideFactor = this.strides.softNext();
                            }
                            if (subChannels <= this.subFactor) {
                                throw new IllegalArgumentException("Subsampling factor must be less than " +
                                        subChannels + " in this configuration.");
                            }
                            // Compute the number of inputs we expect to remain after the subsampling layer.
                            subChannels = (subChannels - this.subFactor) / this.subFactor + 1;
                        }
                        // Compute the default hidden layer width. Note the default is only set if the list
                        // is currently empty.
                        this.denseLayers.setDefault((subChannels + this.labels.size() + 1) / 2);
                    }
                }
                // Save the regularization configuration.
                this.regulizer = new Regularization(this.regMode, this.regFactor);
                // Compute the model file name if it is defaulting.
                if (this.modelName == null)
                    this.modelName = new File(this.modelDir, "model.ser");
                // If the user asked for a configuration file, write it here.
                if (this.parmFile != null) writeParms(this.parmFile);
                // Correct the early stop value.
                if (this.earlyStop == 0) this.earlyStop = this.iterations;
                // Correct the Nesterov learning rate for the weight updater.  The default here is 0.1, not 1e-3
                this.realLearningRate = this.learnRate;
                if (this.weightUpdateMethod == GradientUpdater.Type.NESTEROVS)
                    this.realLearningRate *= 100;
                // Write out the comment.
                if (this.comment != null)
                    log.info("*** {}", this.comment);
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

    /** Write all the parameters to a configuration file.
     *
     * @param outFile	file to be created for future use as a configuration file
     *
     * @throws IOException */
    private void writeParms(File outFile) throws IOException {
        String commentFlag = "";
        PrintWriter writer = new PrintWriter(outFile);
        writer.format("--col %s\t# input column for class name%n", this.labelCol);
        commentFlag = (this.otherMode ? "" : "# ");
        writer.format("%s--other\t# indicates class 0 is a negative condition; per-class accuracies will be displayed%n",
                commentFlag);
        writer.format("--iter %d\t# number of training iterations per batch%n", this.iterations);
        writer.format("--batchSize %d\t# size of each training batch%n", this.batchSize);
        writer.format("--testSize %d\t# size of the testing set, taken from the beginning of the file%n", this.testSize);
        String functions = Stream.of(Type.values()).map(Type::name).collect(Collectors.joining(", "));
        writer.format("## Valid training methods are %s.%n", functions);
        writer.format("--method %s\t# training set processing method%n", this.method.toString());
        writer.format("--earlyStop %d\t# early-stop useless-iteration limit%n", this.earlyStop);
        if (this.denseLayers.isEmpty())
            writer.format("# --widths %d\t# configure number and widths of hidden layers%n", this.reader.getWidth());
        else
            writer.format("--widths %s\t# configure hidden layers%n", this.denseLayers.original());
        functions = Stream.of(Regularization.Mode.values()).map(Regularization.Mode::name).collect(Collectors.joining(", "));
        writer.format("## Valid regularization modes are %s.%n", functions);
        writer.format("--regMode %s\t# regularization mode%n", this.regMode);
        writer.format("--regFactor %g\t# regularization coefficient/factor%n", this.regFactor);
        if (this.maxBatches == Integer.MAX_VALUE)
            writer.println("# --maxBatches 10\t# limit the number of input batches");
        else
            writer.format("--maxBatches %d\t# maximum number of input batches to process%n", this.maxBatches);
        writer.format("--learnRate %e\t# weight learning rate%n", this.learnRate);
        writer.format("--updateRate %g\t# bias update coefficient%n", this.biasRate);
        writer.format("--seed %d\t# random number initialization seed%n", this.seed);
        functions = Stream.of(LossFunction.values()).map(LossFunction::name).collect(Collectors.joining(", "));
        writer.format("## Valid loss functions are %s.%n", functions);
        writer.format("--lossFun %s\t# loss function for scoring output%n", this.lossFunction);
        functions = Stream.of(WeightInit.values()).map(WeightInit::name).collect(Collectors.joining(", "));
        writer.format("## Valid weight initializations are %s.%n", functions);
        writer.format("--weight %s\t# weight initialization method%n", this.weightInitMethod.toString());
        functions = Stream.of(Activation.values()).map(Activation::name).collect(Collectors.joining(", "));
        writer.format("## Valid activation functions are %s.%n", functions);
        writer.format("--init %s\t# initial activation function%n", this.initActivationType.toString());
        writer.format("--activation %s\t# hidden layer activation function%n", this.activationType.toString());
        functions = Stream.of(GradientNormalization.values()).map(GradientNormalization::name).collect(Collectors.joining(", "));
        writer.format("## Valid gradient normalizations are %s.%n", functions);
        writer.format("--gradNorm %s\t# gradient normalization strategy%n", this.gradNorm.toString());
        commentFlag = (this.batchNormFlag ? "" : "# ");
        writer.format("%s--batch\t# use a batch normalization layer%n", commentFlag);
        commentFlag = (this.convolutions.isEmpty() ? "# " : "");
        writer.format("%s--cnn %s\t# convolution kernel sizes%n", commentFlag, this.convolutions.original());
        writer.format("%s--filters %s\t# number of convolution filters to try%n", commentFlag, this.filterSizes.original());
        writer.format("%s--sub %d\t# subsampling factor%n", commentFlag, this.subFactor);
        writer.format("%s--strides %s\t# stride to use for convolution layer%n", commentFlag, this.strides.original());
        functions = Stream.of(GradientUpdater.Type.values()).map(GradientUpdater.Type::name).collect(Collectors.joining(", "));
        writer.format("## Valid updater methods are %s.%n", functions);
        writer.format("--updater %s\t# weight gradient updater method (uses learning rate)%n", this.weightUpdateMethod.toString());
        writer.format("--bUpdater %s\t# bias gradient updater method (uses update rate)%n", this.biasUpdateMethod.toString());
        writer.format("--name %s\t# model file name%n", this.modelName);
        commentFlag = (this.rawMode ? "" : "# ");
        writer.format("%s--raw\t# suppress input normalization%n", commentFlag);
        if (this.comment == null)
            writer.println("# --comment The comment appears in the trial log.");
        else
            writer.format("--comment %s%n", this.comment);
        writer.close();

    }

    public void run() {
        try {
            // Get the testing set and compute the normalizer.
            log.info("Reading testing set (size = {}).", this.testSize);
            this.reader.setBatchSize(this.testSize);
            this.testingSet = this.reader.next();
            if (! this.reader.hasNext())
                log.warn("Training set contains only test data. Batch size = {} but {} records in input.",
                        this.testSize, this.testingSet.numExamples());
            DataNormalization normalizer = null;
            if (! this.rawMode) {
                // Here the model must be normalized.
                log.info("Normalizing data using testing set.");
                normalizer = new NormalizerStandardize();
                normalizer.fit(this.testingSet);
                normalizer.transform(this.testingSet);
                reader.setNormalizer(normalizer);
            }
            // Now we build the model configuration.
            int inWidth = this.reader.getWidth();
            log.info("Building model configuration with input width {} and {} channels.",
                    inWidth, this.channelCount);
            NeuralNetConfiguration.ListBuilder configuration = new NeuralNetConfiguration.Builder()
                    .seed(this.seed)
                    .activation(activationType)
                    .weightInit(WeightInit.XAVIER)
                    .biasUpdater(GradientUpdater.create(this.biasUpdateMethod, this.biasRate))
                    .updater(GradientUpdater.create(this.weightUpdateMethod, this.realLearningRate))
                    .gradientNormalization(this.gradNorm).list();
            // Compute the input type.
            CnnToFeedForwardPreProcessor reshaper = null;
            InputType inputShape = InputType.feedForward(inWidth);
            if (this.channelMode) {
                inputShape = InputType.convolutional(1, inWidth, this.channelCount);
            } else if (! this.convolutions.isEmpty()) {
                inputShape = InputType.convolutionalFlat(1, inWidth, 1);
            }
            configuration.setInputType(inputShape);
            if (! this.convolutions.isEmpty()) {
                // Create the convolution layers.  For the first layer, the channel depth is the number
                // of channels and the output size is the first filter size.
                int depth = this.channelCount;
                int convOut = this.filterSizes.first();
                int strideFactor = this.strides.first();
                for (int convKernel : this.convolutions) {
                    log.info("Creating convolution layer with {} inputs.", inWidth);
                    configuration.layer(new ConvolutionLayer.Builder().activation(this.initActivationType)
                            .nIn(depth).nOut(convOut).kernelSize(1, convKernel)
                            .stride(1, strideFactor).build());
                    // There is one output column for each examined kernel.  Each column has
                    // one value per filter in its vector.
                    inWidth = (inWidth - convKernel) / strideFactor + 1;
                    // The new depth is the number of filters.
                    depth = convOut;
                    convOut = this.filterSizes.softNext();
                    strideFactor = this.strides.softNext();
                }
                if (subFactor > 1) {
                    log.info("Creating subsampling layer with {} inputs.", inWidth);
                    configuration.layer(new SubsamplingLayer.Builder()
                            .kernelSize(1, this.subFactor)
                            .stride(1, this.subFactor).build());
                    // Reduce the input size by the subsampling factor.
                    inWidth = (inWidth - this.subFactor) / this.subFactor + 1;
                }
                // Factor the filter count into the input width because it will be
                // flattened for the dense layer.
                inWidth *= convOut;
            } else {
                // Not convolution, so our input layer is dense.
                if (this.channelMode) {
                    // We have a 2D shape, and no automatic input type to flatten it, so we need
                    // to multiply the input width by the number of channels and set up a flattener.
                    reshaper = new CnnToFeedForwardPreProcessor(1, inWidth, this.channelCount);
                    inWidth *= this.channelCount;
                }
                log.info("Creating feed-forward input layer with width {}.", inWidth);
                configuration.layer(new DenseLayer.Builder().activation(this.initActivationType)
                        .nIn(inWidth).nOut(inWidth)
                        .build());
            }
            // Add batch normalization if desired.
            if (this.batchNormFlag) {
                log.info("Adding batch normalization layer.");
                configuration.layer(new BatchNormalization.Builder().build());
            }
            // Compute the default width for the hidden layer.
            int outputCount = this.labels.size();
            int outWidth;
            // Compute the hidden layers.
            for (int layerSize : this.denseLayers) {
                // The new layer goes in here.
                Layer newLayer;
                if (layerSize == 0) {
                    // Here we have an ElementWiseMultiplicationLayer.
                    outWidth = inWidth;
                    ElementWiseMultiplicationLayer.Builder builder = new ElementWiseMultiplicationLayer.Builder()
                            .nIn(inWidth).nOut(outWidth);
                    newLayer = builder.build();
                    log.info("Creating element-wise mulitplication layer with width {}.", inWidth);
                } else {
                    // Here we have a DenseLayer.
                    outWidth = layerSize;
                    log.info("Creating hidden layer with input width {} and {} outputs.", inWidth, outWidth);
                    DenseLayer.Builder builder = new DenseLayer.Builder().nIn(inWidth).nOut(outWidth);
                    // Do the regularization.
                    this.regulizer.apply(builder);
                    // Build the layer.
                    newLayer = builder.build();
                }
                // Add the layer.
                configuration.layer(newLayer);
                // Set up for the next one.
                inWidth = outWidth;
            }
            // Add the output layer.
            Activation outActivation = (this.lossFunction == LossFunctions.LossFunction.XENT ?
                    Activation.SIGMOID : Activation.SOFTMAX);
            log.info("Creating output layer with input width {} and {} outputs.", inWidth, outputCount);
            configuration.layer(new OutputLayer.Builder(this.lossFunction)
                            .activation(outActivation)
                            .nIn(inWidth).nOut(outputCount).build());
            // Add the preprocessor if we need one.
            if (reshaper != null)
                configuration.inputPreProcessor(0, reshaper);
            // Here we create the model itself.
            log.info("Creating model.");
            MultiLayerNetwork model = new MultiLayerNetwork(configuration.build());
            model.init();
            // Now  we train the model.
            this.reader.setBatchSize(this.batchSize);
            long start = System.currentTimeMillis();
            Trainer myTrainer = Trainer.create(this.method, this, log);
            RunStats runStats = myTrainer.trainModel(model, this.reader, testingSet);
            model = runStats.bestModel;
            String minutes = DurationFormatUtils.formatDuration(System.currentTimeMillis() - start, "mm:ss");
            // Here we save the model.
            if (! runStats.isErrorStop()) {
                log.info("Saving model to {}.", this.modelName);
                ModelSerializer.writeModel(model, this.modelName, true, normalizer);
            }
            // Display the configuration.
            StringBuilder parms = new StringBuilder();
            parms.append(String.format(
                            "%n=========================== Parameters ===========================%n" +
                            "     iterations  = %12d, batch size    = %12d%n" +
                            "     test size   = %12d, seed number   = %12d%n" +
                            "     subsampling = %12d%n" +
                            "     --------------------------------------------------------%n" +
                            "     Training set strategy is %s.%n" +
                            "     Regularization method is %s.%n" +
                            "     Gradient normalization strategy is %s.%n" +
                            "     Bias update method is %s with coefficient %g.%n" +
                            "     Weight initialization method is %s.%n" +
                            "     Weight update method is %s with learning rate %g.%n" +
                            "     Input layer activation type is %s.%n" +
                            "     Hidden layer activation type is %s.%n" +
                            "     Output layer activation type is %s.%n" +
                            "     Output layer loss function is %s.%n" +
                            "     %s minutes to run %d %s (best was %d), with %d score bounces.%n" +
                            "     %d models saved.",
                           this.iterations, this.batchSize, this.testSize, this.seed,
                           this.subFactor, this.method.toString(), this.regulizer,
                           this.gradNorm.toString(), this.biasUpdateMethod.toString(), this.biasRate,
                           this.weightInitMethod.toString(), this.weightUpdateMethod.toString(), this.learnRate,
                           this.initActivationType.toString(), this.activationType.toString(),
                           outActivation.toString(), this.lossFunction.toString(),
                           minutes, runStats.getEventCount(), myTrainer.eventsName(),
                           runStats.getBestEvent(), runStats.getBounceCount(),
                           runStats.getSaveCount()));
            if (! this.convolutions.isEmpty()) {
                parms.append(String.format("%n     Convolution layers used with kernel sizes %s", this.convolutions));
                parms.append(String.format("%n     Convolutions used filter sizes %s and strides %s.",
                        this.filterSizes, this.strides));
            }
            if (this.batchNormFlag)
                parms.append(String.format("%n     Batch normalization applied."));
            parms.append(String.format("%n     Hidden layer configuration is %s.", this.denseLayers));
            if (this.rawMode)
                parms.append(String.format("%n     Data normalization is turned off."));
            if (this.channelMode)
                parms.append(String.format("%n     Input uses channel vectors."));
            if (runStats.isErrorStop()) {
                parms.append(String.format("%n%nMODEL FAILED DUE TO OVERFLOW OR UNDERFLOW."));
                this.bestAccuracy = 0;
            } else {
                // Now we evaluate the model on the test set: compare the output to the actual
                // values.
                Evaluation eval = Trainer.evaluateModel(model, this.testingSet, this.labels);
                // Output the evaluation.
                parms.append(eval.stats());
                ConfusionMatrix<Integer> matrix = eval.getConfusion();
                // This last thing is the table of scores for each prediction.  This only makes sense if we have
                // an "other" mode.
                if (this.otherMode) {
                    int actualNegative = matrix.getActualTotal(0);
                    if (actualNegative == 0) {
                        parms.append(String.format("%nNo \"%s\" results were found.%n", this.labels.get(0)));
                    } else {
                        double specificity = ((double) matrix.getCount(0, 0)) / actualNegative;
                        parms.append(String.format("%nModel specificity is %11.4f.%n", specificity));
                    }
                    parms.append(String.format("%n%-11s %11s %11s %11s %11s%n", "class", "accuracy", "sensitivity", "precision", "fallout"));
                    parms.append(StringUtils.repeat('-', 59));
                    // The classification accuracy is 1 - (false negative + false positive) / total,
                    // sensitivity is true positive / actual positive, precision is true positive / predicted positive,
                    // and fall-out is false positive / actual negative.
                    for (int i = 1; i < this.labels.size(); i++) {
                        String label = this.labels.get(i);
                        double accuracy = 1 - ((double) (matrix.getCount(0, i) + matrix.getCount(i,  0))) / this.testSize;
                        String sensitivity = formatRatio(matrix.getCount(i, i), matrix.getActualTotal(i));
                        String precision = formatRatio(matrix.getCount(i, i), matrix.getPredictedTotal(i));
                        String fallout = formatRatio(matrix.getCount(0,  i), actualNegative);
                        parms.append(String.format("%n%-11s %11.4f %11s %11s %11s", label, accuracy, sensitivity, precision, fallout));
                    }
                }
                // Finally, save the accuracy in case SearchProcessor is running us.
                this.bestAccuracy = eval.accuracy();
            }
            // Add the summary.
            parms.append(model.summary(inputShape));
            String report = parms.toString();
            log.info(report);
            writeTrialReport(this.comment, report);
        } catch (Exception e) {
            log.error("Error in training: {}", e.getMessage());
        }
    }

    /**
     * Format a ratio for display in the evaluation metrics matrix.
     *
     * @param count		numerator
     * @param total		denominator
     *
     * @return a formatted fraction, or an empty string if the denominator is 9
     */
    private static String formatRatio(int count, int total) {
        String retVal = "";
        if (total > 0) {
            retVal = String.format("%11.4f", ((double) count) / total);
        }
        return retVal;
    }

    /**
     * Write a report to the trial log.
     *
     * @param label	heading comment, if any
     * @param report	text of the report to write, with internal new-lines
     *
     * @throws IOException
     */
    public void writeTrialReport(String label, String report) throws IOException {
        // Open the trials log in append mode and write the information about this run.
        File trials = new File(modelDir, "trials.log");
        PrintWriter trialWriter = new PrintWriter(new FileWriter(trials, true));
        trialWriter.println("******************************************************************");
        if (label != null)
            trialWriter.print(label);
        trialWriter.println(report);
        trialWriter.close();
    }

    /**
     * @return the accuracy of the best epoch
     */
    public double getAccuracy() {
        return bestAccuracy;
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
     * @return the label names for this model
     */
    public List<String> getLabels() {
        return this.labels;
    }

    /**
     * @return the early-stop limit
     */
    public int getEarlyStop() {
        return this.earlyStop;
    }
}
