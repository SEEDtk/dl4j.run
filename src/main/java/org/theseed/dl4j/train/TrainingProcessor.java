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
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.commons.lang3.StringUtils;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.dropout.Dropout;
import org.deeplearning4j.nn.conf.dropout.GaussianDropout;
import org.deeplearning4j.nn.conf.dropout.IDropout;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.dl4j.ChannelDataSetReader;
import org.theseed.dl4j.TabbedDataSetReader;
import org.theseed.utils.ICommand;


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
 * 		default is 0 (last column)
 * -n	number of iterations to run on each batch; the default is 1000
 * -b	size of each batch; the default is 500
 * -t	size of the testing set, which is the first batch read and is used to compute
 * 		normalization; the default is 2000
 * -w	the width of each hidden layer; this is a comma-delimited list with one number per
 * 		layer, in order; if no hidden layers are specified, a default hidden layer whose
 * 		width is the mean of the input and output widths will be created
 * -g	Gaussian dropout rate used to prevent overfitting-- a higher value makes the model
 * 		less sensitive and a lower one makes it more sensitive; the default is 0.3
 * -u	bias updater coefficient, used to determine the starting speed of the model-- a
 * 		higher value makes the learning faster, a lower value makes it more accurate; the
 * 		default is 0.2
 * -x	maximum number of batches to process (not including the testing set); use this to
 * 		run smaller input sets for debugging the tuning parameters; the default is MAX_INTEGER,
 * 		indicating all batches should be run
 * -s	seed value to use for random number generation; the default is to use the last
 * 		20 bits of the current time
 * -d	number of hidden layers to use (depth); the default is 1
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
 * --l2			if nonzero, then l2 regularization is used instead of gaussian dropout; the
 * 				value should be the regularization parameter; the default is 0
 * --drop		if specified, then regular dropout is used instead of gaussian dropout
 * --cnn		if specified, then the input layer will be a convolution layer with the
 * 				specified kernel width; otherwise, it will be a standard dense layer
 * --init		activation type for the input layer; the default is "tanh"
 *
 * For a convolution input layer, the following additional parameters are used.
 *
 * --sub		indicates a subsampling layer will be used with the indicated kernel/stride size to
 * 				reduce the output from the convolution layer; the default is 1, meaning no subsampling
 * --filters	specifies the number of filters to use during convolution; each filter represents a pass
 * 				over the input with different trial parameters; the output size is multiplied by the
 * 				number of filters
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
    List<String> labels;
    /** testing set */
    DataSet testingSet;
    /** number of input channels */
    int channelCount;
    /** TRUE if we have channel input */
    boolean channelMode;

    /** logging facility */
    private static Logger log = LoggerFactory.getLogger(TrainingProcessor.class);

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
            usage="width of each hidden layer")
    private String layerWidths;

    /** dropout rate to prevent overfitting */
    @Option(name="-g", aliases={"--gaussRate"}, metaVar="0.5",
            usage="Gaussian dropout rate")
    private double gaussRate;

    /** linear dropout rate; if not zero, overrides gaussian dropout */
    @Option(name="--drop", usage="normal (non-gaussian) dropout mode")
    private double normalDrop;

    /** l2 regularization option */
    @Option(name="--l2", metaVar="0.2", usage="l2 regularization parameter (replaces gaussian dropout if nonzero)")
    private double l2Parm;

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

    /** convolution mode */
    @Option(name="--cnn", metaVar="3", usage="convolution mode, specifying kernel size")
    private int convolution;

    /** subsampling layer */
    @Option(name="--sub", metaVar="2", usage="kernel/stride of subsampling layer (if any)")
    private int subFactor;

    /** convolution output */
    @Option(name="--filters", metaVar="3", usage="number of trial filters to use for convolution")
    private int convOut;

    /** comma-delimited list of metadata column names */
    @Option(name="--meta", metaVar="name,date", usage="comma-delimited list of metadata columns")
    private String metaCols;

    /**

    /** model directory */
    @Argument(index=0, metaVar="modelDir", usage="model directory", required=true)
    private File modelDir;


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
        this.labelCol = "0";
        this.iterations = 1000;
        this.batchSize = 500;
        this.testSize = 2000;
        this.layerWidths = "";
        this.gaussRate = 0.3;
        this.biasRate = 0.2;
        this.learnRate = 1e-3;
        this.maxBatches = Integer.MAX_VALUE;
        this.lossFunction = LossFunctions.LossFunction.MCXENT;
        this.seed = (int) (System.currentTimeMillis() & 0xFFFFF);
        this.trainingFile = null;
        this.l2Parm = 0;
        this.activationType = Activation.RELU;
        this.initActivationType = Activation.TANH;
        this.gradNorm = GradientNormalization.None;
        this.normalDrop = 0.0;
        this.convolution = 0;
        this.channelMode = false;
        this.subFactor = 1;
        this.convOut = 1;
        this.metaCols = "";
        this.channelCount = 1;
        this.parmFile = null;
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

                        // Finally, verify that the subfactor is in range.
                        if (this.convolution > 0 && this.subFactor > 1) {
                            int subChannels = this.reader.getWidth() - this.convolution + 1;
                            if (subChannels <= this.subFactor) {
                                throw new IllegalArgumentException("Subsampling factor must be less than " +
                                        subChannels + " in this configuration.");
                            }
                        }
                    }
                }
                // If the user asked for a configuration file, write it here.
                if (this.parmFile != null) writeParms(this.parmFile);
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
        writer.format("--col %s\t\t# input column for class name%n", this.labelCol);
        writer.format("--iter %d\t\t# number of training iterations per batch%n", this.iterations);
        writer.format("--batchSize %d\t\t# size of each training batch%n", this.batchSize);
        writer.format("--testSize %d\t\t# size of the testing set, taken from the beginning of the file%n", this.testSize);
        if (this.layerWidths.isEmpty())
            writer.format("# --widths %d\t\t# configure number and widths of hidden layers%n", this.reader.getWidth());
        else
            writer.format("--widths %s\t\t# configure hidden layers%n", this.layerWidths);
        commentFlag = ((this.normalDrop > 0 || this.l2Parm > 0) ? "#" : "");
        writer.format("%s--gaussRate %g\t\t# gaussian dropout rate%n", commentFlag, this.gaussRate);
        commentFlag = ((this.l2Parm > 0) ? "# " : "");
        writer.format("%s--drop %g\t\t# linear dropout rate (overrides gaussian)%n", commentFlag, this.normalDrop);
        commentFlag = ((this.l2Parm == 0) ? "# " : "");
        writer.format("%s--l2 %g\t\t# L2 regularization factor (overrides dropouts)%n", commentFlag, this.l2Parm);
        if (this.maxBatches == Integer.MAX_VALUE)
            writer.println("# --maxBatches 10\t\t# limit the number of input batches");
        else
            writer.format("--maxBatches %d\t\t# maximum number of input batches to process%n", this.maxBatches);
        writer.format("--learnRate %e\t\t# learning rate%n", this.learnRate);
        writer.format("--updateRate %g\t\t# bias updater coefficient%n", this.biasRate);
        writer.format("--seed %d\t\t# random number initialization seed%n", this.seed);
        String functions = Stream.of(LossFunction.values()).map(LossFunction::name).collect(Collectors.joining(", "));
        writer.format("## Valid loss functions are %s.%n", functions);
        writer.format("--lossFun %s\t\t# loss function for scoring output%n", this.lossFunction);
        functions = Stream.of(Activation.values()).map(Activation::name).collect(Collectors.joining(", "));
        writer.format("## Valid activation functions are %s.%n", functions);
        writer.format("--init %s\t\t# initial activation function%n", this.initActivationType.toString());
        writer.format("--activation %s\t\t# hidden layer activation function%n", this.activationType.toString());
        functions = Stream.of(GradientNormalization.values()).map(GradientNormalization::name).collect(Collectors.joining(", "));
        writer.format("## Valid gradient normalizations are %s.%n", functions);
        writer.format("--gradNorm %s\t\t# gradient normalization strategy%n", this.gradNorm.toString());
        commentFlag = ((this.convolution > 0) ? "" : "# ");
        writer.format("%s--cnn %d\t\t# convolution kernel size%n", commentFlag, this.convolution);
        writer.format("%s--filters %d\t\t# number of convolution filters to try%n", commentFlag, this.convOut);
        writer.format("%s--sub %d\t\t# subsampling factor%n", commentFlag, this.subFactor);
        commentFlag = (this.rawMode ? "" : "# ");
        writer.format("%s--raw\t\t# suppress input normalization%n", commentFlag);
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
                    .biasUpdater(new Sgd(this.biasRate))
                    .updater(new Adam(this.learnRate))
                    .gradientNormalization(this.gradNorm).list();
            // Compute the input type.
            CnnToFeedForwardPreProcessor reshaper = null;
            InputType inputShape = InputType.feedForward(inWidth);
            if (this.channelMode) {
                inputShape = InputType.convolutional(1, inWidth, this.channelCount);
            } else if (this.convolution > 0) {
                inputShape = InputType.convolutionalFlat(1, inWidth, 1);
            }
            configuration.setInputType(inputShape);
            // This variable will track the number of hidden layers we need.
            // We subtract one for every extra input layer.
            int layers = 0;
            if (this.convolution > 0) {
                log.info("Creating convolution layer with {} inputs.", inWidth);
                configuration.layer(new ConvolutionLayer.Builder().activation(this.initActivationType)
                        .nIn(this.channelCount).nOut(this.convOut).kernelSize(1, this.convolution).build());
                // There is one output column for each examined kernel.  Each column has
                // one value per filter in its vector.
                inWidth = (inWidth - this.convolution + 1);
                layers++;
                if (subFactor > 1) {
                    log.info("Creating subsampling layer with {} inputs.", inWidth);
                    configuration.layer(new SubsamplingLayer.Builder()
                            .kernelSize(1, this.subFactor)
                            .stride(1, this.subFactor).build());
                    // Reduce the input size by the subsampling factor.
                    inWidth = (inWidth - this.subFactor) / this.subFactor + 1;
                    layers++;
                }
                // Factor the filter count into the input width because it will be
                // flattened for the dense layer.
                inWidth *= this.convOut;
            } else if (this.channelMode) {
                // Not convolution, but we need to multiply the input width by the
                // number of channels and set up a flattener.
                reshaper = new CnnToFeedForwardPreProcessor(1, inWidth, this.channelCount);
                inWidth *= this.channelCount;
            }
            // Compute the default width for the hidden layer.
            int outputCount = this.labels.size();
            int outWidth = (inWidth + outputCount) / 2;
            if (outWidth < 3) outWidth = 3;
            Queue<Integer> hiddenLayers = new LinkedList<Integer>();
            if (this.layerWidths.isEmpty() && layers == 0) this.layerWidths = String.valueOf(outWidth);
            if (! this.layerWidths.isEmpty()) {
                Arrays.stream(this.layerWidths.split(",")).mapToInt(Integer::parseInt)
                         .forEachOrdered(hiddenLayers::add);
            }
            log.info("Creating feed-forward layer with width {}.", inWidth);
            configuration.layer(new DenseLayer.Builder().activation(this.initActivationType)
                    .nIn(inWidth).nOut(inWidth)
                    .build());
            // Compute the hidden layers.
            while (hiddenLayers.size() > 0) {
                outWidth = hiddenLayers.remove();
                log.info("Creating hidden layer with input width {} and {} outputs.", inWidth, outWidth);
                DenseLayer.Builder builder = new DenseLayer.Builder().nIn(inWidth).nOut(outWidth);
                // Do the regularization.
                if (this.l2Parm > 0) {
                    builder.l2(this.l2Parm);
                } else {
                    IDropout dropOut;
                    if (this.normalDrop > 0) {
                        dropOut = new Dropout(this.normalDrop);
                    } else {
                        dropOut = new GaussianDropout(this.gaussRate);
                    }
                    builder.dropOut(dropOut);
                }
                // Build and add the layer.
                configuration.layer(builder.build());
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
            // Now  we begin running the model.
            boolean errorStop = false;
            int batchCount = 0;
            int bounceCount = 0;
            double oldScore = Double.MAX_VALUE;
            this.reader.setBatchSize(this.batchSize);
            while (reader.hasNext() && batchCount < this.maxBatches && ! errorStop) {
                batchCount++;
                log.info("Reading data batch {}.", batchCount);
                DataSet trainingData = reader.next();
                for(int i=0; i < iterations; i++ ) {
                    model.fit(trainingData);
                }
                double newScore = model.score();
                if (oldScore < newScore) bounceCount++;
                oldScore = newScore;
                log.info("Score at end of batch {} is {}.", batchCount, newScore);
                if (! Double.isFinite(newScore)) {
                    log.error("Overflow/Underflow in gradient processing.  Model abandoned.");
                    errorStop = true;
                }
            }
            // Here we save the model.
            if (! errorStop) {
                File saveFile = new File(this.modelDir, "model.ser");
                log.info("Saving model to {}.", saveFile);
                ModelSerializer.writeModel(model, saveFile, true, normalizer);
            }
            INDArray output = model.output(this.testingSet.getFeatures());
            // Display the configuration.
            String regularization;
            double regFactor;
            if (this.l2Parm > 0) {
                regularization = "L2";
                regFactor = this.l2Parm;
            } else if (this.normalDrop > 0) {
                regularization = "Linear dropout";
                regFactor = this.normalDrop;
            } else {
                regularization = "Gauss dropout";
                regFactor = this.gaussRate;
            }
            String parms = String.format("%n=========================== Parameters ===========================%n" +
                    "     iterations  = %12d, batch size    = %12d%n" +
                    "     test size   = %12d, seed number   = %12d%n" +
                    "     learn rate  = %12e, bias rate     = %12g%n" +
                    "     convolution = %12d, conv. filters = %12d%n" +
                    "     subsampling = %12d%n" +
                    "     --------------------------------------------------------%n" +
                    "     Regularization method is %s with factor %g.%n" +
                    "     Gradient normalization strategy is %s.%n" +
                    "     Input layer activation type is %s.%n" +
                    "     Hidden layer activation type is %s.%n" +
                    "     Output layer activation type is %s.%n" +
                    "     Output layer loss function is %s.%n" +
                    "     %d total batches run with %d score bounces.",
                   this.iterations, this.batchSize, this.testSize, this.seed,
                   this.learnRate, this.biasRate, this.convolution, this.convOut,
                   this.subFactor, regularization, regFactor, this.gradNorm.name(),
                   this.initActivationType.name(), this.activationType.name(),
                   outActivation.name(), this.lossFunction.name(),
                   batchCount, bounceCount);
            if (! this.layerWidths.isEmpty()) {
                 String layerConfig = this.layerWidths.replaceAll(",", ", ");
                 parms += String.format("%n     Layer configuration is %s", layerConfig);
            }
            if (this.rawMode)
                parms += String.format("%n     Normalization is turned off.");
            if (this.channelMode)
                parms += String.format("%n     Input uses channel vectors.");
            log.info(parms);
            String statDisplay;
            if (errorStop) {
                statDisplay = String.format("%n%nMODEL FAILED DUE TO OVERFLOW OR UNDERFLOW.");
            } else {
                //evaluate the model on the test set: compare the output to the actual
                Evaluation eval = new Evaluation(this.labels);
                eval.eval(this.testingSet.getLabels(), output);
                // Output the evaluation.  Note we are more aggressive about allowing the confusion matrix than
                // the default process is.
                boolean showConfusion = (this.labels.size() < 50);
                statDisplay = eval.stats(false, showConfusion);
            }
            // Add the summary.
            statDisplay += model.summary(inputShape);
            log.info(statDisplay);
            // Open the trials log in append mode and write the information about this run.
            File trials = new File(modelDir, "trials.log");
            PrintWriter trialWriter = new PrintWriter(new FileWriter(trials, true));
            trialWriter.print(parms);
            trialWriter.println(statDisplay);
            trialWriter.close();
        } catch (IOException e) {
            log.error(e.getMessage());
        }
    }
}
