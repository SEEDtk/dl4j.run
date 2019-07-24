/**
 *
 */
package org.theseed.dl4j.train;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Scanner;

import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.dropout.Dropout;
import org.deeplearning4j.nn.conf.dropout.GaussianDropout;
import org.deeplearning4j.nn.conf.dropout.IDropout;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
//import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.dl4j.App;
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
 * -w	the width of each layer; the default is the mean of the number of feature sensors
 * 		and the number of classes
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
 * -m	slope of the hidden layer width; the last hidden layer has the layer width; each
 * 		layer above it has this many additional nodes; the default is 0
 * -r	learning rate; this should be between 1e-1 and 1e-6; the default is 1e-3
 * -z	gradient normalization strategy; the default is "none"
 *
 * --raw	if specified, the data is not normalized
 * --l2		if nonzero, then l2 regularization is used instead of gaussian dropout; the
 * 			value should be the regularization parameter; the default is 0
 * --drop	if specified, then regular dropout is used instead of gaussian dropout
 * --cnn	if specified, then the input layer will be a one-dimensional convolution
 * 			layer with the specified kernel width; otherwise, it will be a standard
 * 			dense layer
 *
 * @author Bruce Parrello
 *
 */
public class TrainingProcessor implements ICommand {

    // FIELDS
    /** input file reader */
    private TabbedDataSetReader reader;
    /** array of labels */
    ArrayList<String> labels;
    /** testing set */
    DataSet testingSet;

    /** logging facility */
    private static Logger log = LoggerFactory.getLogger(App.class);

    // COMMAND LINE

    /** help option */
    @Option(name="-h", aliases={"--help"}, help=true)
    private boolean help;

    /** name or index of the label column */
    @Option(name="-c", aliases={"--col"}, metaVar="0",
            usage="input column containing class")
    private String labelCol;

    /** name or index of the label column */
    @Option(name="-n", aliases={"--iter"}, metaVar="1000",
            usage="number of iterations per batch")
    private int iterations;

    /** size ofeach input batch */
    @Option(name="-b", aliases={"--batchSize"}, metaVar="1000",
            usage="size of each input batch")
    private int batchSize;

    /** size of testing set */
    @Option(name="-t", aliases={"--testSize"}, metaVar="1000",
            usage="size of the testing set")
    private int testSize;

    /** number of nodes in each middle layer */
    @Option(name="-w", aliases={"--width"}, metaVar="7",
            usage="width of each neural net layer")
    private int layerWidth;

    /** number of nodes to add going from output layer to input layer */
    @Option(name="-m", aliases= {"--slope"}, metaVar="1",
            usage="slope of the layer width going from output to input")
    private int layerSlope;

    /** dropout rate to prevent overfitting */
    @Option(name="-g", aliases={"--gaussRate"}, metaVar="0.5",
            usage="Gaussian dropout rate")
    private double gaussRate;

    /** TRUE for linear dropout mode */
    @Option(name="--drop", usage="normal (non-gaussian) dropout mode")
    private boolean normalDrop;

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

    /** number of hidden layers */
    @Option(name="-d", aliases={"--layers", "--depth"}, metaVar="1",
            usage="number of hidden layers (depth)")
    private int layers;

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
//	@Option(name="--cnn", metaVar="3", usage="convolution mode, specifying kernel size")
    private int convolution;

    /** model directory */
    @Argument(index=0, metaVar="modelDir", usage="model directory", required=true)
    private File modelDir;

    /**

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
        this.layerWidth = 0;
        this.layerSlope = 0;
        this.gaussRate = 0.3;
        this.biasRate = 0.2;
        this.learnRate = 1e-3;
        this.maxBatches = Integer.MAX_VALUE;
        this.lossFunction = LossFunctions.LossFunction.MCXENT;
        this.seed = (int) (System.currentTimeMillis() & 0xFFFFF);
        this.layers = 1;
        this.trainingFile = null;
        this.l2Parm = 0;
        this.activationType = Activation.RELU;
        this.gradNorm = GradientNormalization.None;
        this.normalDrop = false;
        this.convolution = 0;
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
                    File labelFile = new File(this.modelDir, "labels.txt");
                    if (! labelFile.exists()) {
                        throw new FileNotFoundException("Label file not found in " + this.modelDir + ".");
                    } else {
                        Scanner labelsIn = new Scanner(labelFile);
                        this.labels = new ArrayList<String>();
                        while (labelsIn.hasNext()) {
                            this.labels.add(labelsIn.nextLine());
                        }
                        labelsIn.close();
                        log.info("{} labels read from label file.", this.labels.size());
                        // Finally, we initialize the input to get the label column handled.
                        if (this.trainingFile == null) {
                            this.trainingFile = new File(this.modelDir, "training.tbl");
                            if (! this.trainingFile.exists())
                                throw new FileNotFoundException("Training file " + this.trainingFile + " not found.");
                        }
                        this.reader = new TabbedDataSetReader(this.trainingFile, this.labelCol, this.labels);
                        // Now that we know the number of labels, we can default the layer width.
                        // We set it to the midpoint between the inputs and the outputs (rounded up),
                        // with a minimum of 3 ( since 2 crashes the engine).
                        if (this.layerWidth == 0) {
                            this.layerWidth = (this.labels.size() + this.reader.getWidth() + 1) / 2;
                            if (this.layerWidth == 2) this.layerWidth = 3;
                        }
                    }
                }
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

    public void run() {
        try {
            // Get the testing set and compute the normalizer.
            log.info("Reading testing set (size = {}).", this.testSize);
            this.reader.setBatchSize(this.testSize);
            this.testingSet = this.reader.next();
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
            log.info("Building model configuration with hidden layer width {} and depth {}.",
                    this.layerWidth, this.layers);
            int outWidth = this.layerWidth + (this.layers - 1) * this.layerSlope;
            NeuralNetConfiguration.ListBuilder configuration = new NeuralNetConfiguration.Builder()
                    .seed(this.seed)
                    .activation(activationType)
                    .weightInit(WeightInit.XAVIER)
                    .biasUpdater(new Sgd(this.biasRate))
                    .updater(new Adam(this.learnRate))
                    .gradientNormalization(this.gradNorm).list();
            if (this.convolution == 0) {
                configuration.layer(new DenseLayer.Builder().activation(Activation.TANH)
                        .nIn(this.reader.getWidth()).nOut(outWidth)
                        .build());
            } else {
                throw new IllegalArgumentException("Convolution not working yet.");
//            	configuration.layer(new ConvolutionLayer.Builder().nIn(this.reader.width())
//            			.nOut(outWidth).kernelSize(3, 1).build());
            }
            // Add the hidden layers.
            for (int i = 1; i <= this.layers; i++) {
                log.info("Layer {} width is {}.", i, outWidth);
                int inWidth = outWidth;
                outWidth = inWidth - this.layerSlope;
                DenseLayer.Builder builder = new DenseLayer.Builder().nIn(inWidth).nOut(outWidth);
                // Do the regularization.
                if (this.l2Parm > 0) {
                    builder.l2(this.l2Parm);
                } else {
                    IDropout dropOut;
                    if (this.normalDrop) {
                        dropOut = new Dropout(this.gaussRate);
                    } else {
                        dropOut = new GaussianDropout(this.gaussRate);
                    }
                    builder.dropOut(dropOut);
                }
                // Build and add the layer.
                configuration.layer(builder.build());
            }
            // Add the output layer.
            int outputCount = this.labels.size();
            Activation outActivation = (this.lossFunction == LossFunctions.LossFunction.XENT ?
                    Activation.SIGMOID : Activation.SOFTMAX);
            configuration.layer(this.layers + 1,
                    new OutputLayer.Builder(this.lossFunction)
                            .activation(outActivation)
                            .nIn(outWidth).nOut(outputCount).build());
            // Here we create the model itself.
            log.info("Creating model.");
            MultiLayerNetwork model = new MultiLayerNetwork(configuration.build());
            model.init();
            // Now  we begin running the model.
            int batchCount = 0;
            int bounceCount = 0;
            double oldScore = Double.MAX_VALUE;
            this.reader.setBatchSize(this.batchSize);
            while (reader.hasNext() && batchCount < this.maxBatches) {
                batchCount++;
                log.info("Reading data batch {}.", batchCount);
                DataSet trainingData = reader.next();
                for(int i=0; i < iterations; i++ ) {
                    model.fit(trainingData);
                }
                double newScore = model.score();
                if (oldScore < newScore) bounceCount++;
                oldScore = newScore;
                log.info("Score at end of batch {} is {}.", batchCount, model.score());
            }
            // Here we save the model.
            File saveFile = new File(this.modelDir, "model.ser");
            log.info("Saving model to {}.", saveFile);
            ModelSerializer.writeModel(model, saveFile, true, normalizer);
            INDArray output = model.output(this.testingSet.getFeatures());
            // Display the configuration.
            String regularization;
            double regFactor;
            if (this.l2Parm > 0) {
                regularization = "L2";
                regFactor = this.l2Parm;
            } else {
                regularization = (this.normalDrop ? "Linear dropout" : "Gauss dropout");
                regFactor = this.gaussRate;
            }
            String parms = String.format("%n=========================== Parameters ===========================%n" +
                    "     iterations  = %12d, batch size    = %12d%n" +
                    "     test size   = %12d, layer width   = %12d%n" +
                    "     learn rate  = %12e, bias rate     = %12g%n" +
                    "     seed number = %12d, hidden layers = %12d%n" +
                    "     layer slope = %12d, convolution   = %12d%n" +
                    "     --------------------------------------------------------%n" +
                    "     Regularization method is %s with factor %g.%n" +
                    "     Gradient normalization strategy is %s.%n" +
                    "     Hidden layer activation type is %s.%n" +
                    "     Output layer activation type is %s.%n" +
                    "     Output layer loss function is %s.%n" +
                    "     %d total batches run with %d score bounces.",
                   this.iterations, this.batchSize, this.testSize, this.layerWidth,
                   this.learnRate, this.biasRate, this.seed, this.layers, this.layerSlope,
                   this.convolution, regularization, regFactor, this.gradNorm.name(),
                   this.activationType.name(), outActivation.name(),
                   this.lossFunction.name(), batchCount, bounceCount);
            if (this.rawMode)
               parms += String.format("%nNormalization is turned off.");
            log.info(parms);
            //evaluate the model on the test set: compare the output to the actual
            Evaluation eval = new Evaluation(this.labels);
            eval.eval(this.testingSet.getLabels(), output);
            // Output the evaluation.
            String statDisplay = eval.stats();
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
