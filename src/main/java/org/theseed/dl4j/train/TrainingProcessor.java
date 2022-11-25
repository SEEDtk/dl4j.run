package org.theseed.dl4j.train;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UncheckedIOException;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.commons.text.TextStringBuilder;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.misc.ElementWiseMultiplicationLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.theseed.counters.Shuffler;
import org.theseed.dl4j.CnnToRnnSequencePreprocessor;
import org.theseed.dl4j.DistributedOutputStream;
import org.theseed.dl4j.LossFunctionType;
import org.theseed.dl4j.Regularization;
import org.theseed.dl4j.RnnSequenceToFeedForwardPreProcessor;
import org.theseed.dl4j.TabbedDataSetReader;
import org.theseed.io.LineReader;
import org.theseed.reports.IValidationReport;
import org.theseed.reports.TestValidationReport;
import org.theseed.utils.FloatList;
import org.theseed.utils.IntegerList;
import org.theseed.utils.Parms;

public abstract class TrainingProcessor extends LearningProcessor implements ITrainingProcessor {

    /** underflow error message */
    private static final String UNDERFLOW_ERROR = "MODEL FAILED DUE TO OVERFLOW OR UNDERFLOW.";



    // FIELDS

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
    /** regularization control object */
    private Regularization regulizer;
    /** weights for loss function */
    private FloatList lossWeights;
    /** input shape for model */
    private InputType inputShape;
    /** output layer activation function */
    private Activation outActivation;
    /** result string from most recent training */
    private String resultReport;

    // COMMAND-LINE OPTIONS

    /** parameter dump */
    @Option(name="--parms", usage="write command-line parameters to a configuration file")
    private File parmFile;

    /** \regularization factor to prevent overfitting */
    @Option(name = "--regFactor", metaVar = "0.5", usage = "regularization coefficient/factor")
    private double regFactor;

    /** number of nodes in each middle layer */
    @Option(name = "-w", aliases = {
            "--widths" }, metaVar = "7", usage = "width of each hidden layer (0 for element-wise multiplication")
    private void setLayers(String layerWidths) {
        if (! layerWidths.equalsIgnoreCase("none"))
            this.denseLayers = new IntegerList(layerWidths);
    }

    /** regularization mode */
    @Option(name = "--regMode", usage = "regularization mode")
    private Regularization.Mode regMode;
    /** bias updater coefficient */
    @Option(name = "-u", aliases = { "--updateRate" }, metaVar = "0.1", usage = "bias updater coefficient")
    private double biasRate;
    /** loss function */
    @Option(name = "-l", aliases = { "--lossFun",
            "--loss" }, metaVar = "mse", usage = "loss function for scoring output layer")
    private LossFunctionType lossFunction;
    /** normalization flag */
    @Option(name = "--raw", usage = "suppress dataset normalization")
    private boolean rawMode;
    /** default activation function */
    @Option(name = "-a", aliases = { "--activation" }, usage = "activation function for hidden layers")
    private Activation activationType;
    /** initial activation function */
    @Option(name = "--init", usage = "activation function for input layer")
    private Activation initActivationType;
    /** learning rate */
    @Option(name = "-r", aliases = { "--learnRate" }, metaVar = "0.1", usage = "learning rate")
    private double learnRate;
    /** gradient normalization strategy */
    @Option(name = "-z", aliases = {
            "--gradNorm" }, metaVar = "RenormalizeL2PerLayer", usage = "gradient normalization strategy")
    private GradientNormalization gradNorm;
    /** use batch normalization */
    @Option(name = "--batch", usage = "use batch normalization before hidden layers")
    private boolean batchNormFlag;
    /** LSTM layers */
    @Option(name = "--lstm", metaVar = "50", usage = "graves LSTM layers, specifying output size of each layer")
    private int lstmLayers;

    /** convolution mode */
    @Option(name = "--cnn", metaVar = "3", usage = "convolution mode, specifying kernel size of each layer")
    private void setConvolution(String cnn) {
        this.convolutions = new IntegerList(cnn);
    }

    /** subsampling layer */
    @Option(name = "--sub", metaVar = "2", usage = "kernel/stride of subsampling layer (if any)")
    private int subFactor;
    /** weight initialization algorithm */
    @Option(name = "--start", usage = "weight initialization strategy")
    private WeightInit weightInitMethod;

    /** convolution output */
    @Option(name = "--filters", metaVar = "3", usage = "number of trial filters to use for each convolution")
    private void setFilters(String filters) {
        this.filterSizes = new IntegerList(filters);
    }

    /** convolution strides */
    @Option(name = "--strides", metaVar = "3,2", usage = "stride to use for each convolution")
    private void setStrides(String strides) {
        this.strides = new IntegerList(strides);
    }

    /** bias updater algorithm */
    @Option(name = "--bUpdater", usage = "bias gradient updater algorithm")
    private GradientUpdater.Type biasUpdateMethod;
    /** weight updater algorithm */
    @Option(name = "--updater", usage = "weight gradient updater algorithm")
    private GradientUpdater.Type weightUpdateMethod;
    /** use balanced layers */
    @Option(name = "--balanced", metaVar = "4", usage = "compute balanced layer widths for the specified number of layers")
    private int balancedLayers;

    /** loss function weights */
    @Option(name = "--weights", metaVar = "1.0,0.5", usage = "comma-delimited list of loss function weights, by label")
    private void setWeights(String weightString) {
        this.lossWeights = new FloatList(weightString);
    }

    /**
     * Set the parameter defaults related to model creation.
     */
    public void setModelDefaults() {
        this.parmFile = null;
        this.denseLayers = new IntegerList();
        this.regFactor = 0.3;
        this.biasRate = 0.2;
        this.learnRate = 1e-3;
        this.trainingFile = null;
        this.regMode = Regularization.Mode.GAUSS;
        this.activationType = Activation.RELU;
        this.initActivationType = Activation.HARDTANH;
        this.gradNorm = GradientNormalization.None;
        this.convolutions = new IntegerList();
        this.lstmLayers = 0;
        this.subFactor = 1;
        this.filterSizes = new IntegerList("1");
        this.strides = new IntegerList("1");
        this.batchNormFlag = false;
        this.weightInitMethod = WeightInit.XAVIER;
        this.biasUpdateMethod = GradientUpdater.Type.NESTEROVS;
        this.weightUpdateMethod = GradientUpdater.Type.ADAM;
        this.balancedLayers = 0;
        this.lossWeights = new FloatList();
    }

    /**
     * Set up the common input parameters.
     *
     * @throws FileNotFoundException
     * @throws IOException
     */
    protected void initializeModelParameters() throws FileNotFoundException, IOException {
        // Check the loss function.
        if (this.lossFunction.isBinaryOnly() && this.getLabels().size() > 2)
            throw new IllegalArgumentException(this.lossFunction + " is for binary classification but there are " +
                    this.getLabels().size() + " classes.");
        if (this.lossWeights.isEmpty()) {
            // Here we need to default the list to the distribution in the testing set.
            log.info("Computing loss function weights from testing set.");
            double[] buffer = new double[this.getLabels().size()];
            INDArray labelSums = this.getTestingSet().getLabels().sum(0);
            double base = labelSums.maxNumber().doubleValue();
            for (int i = 0; i < buffer.length; i++)
                buffer[i] = labelSums.getDouble(i) / base;
            this.lossWeights = new FloatList(buffer);
        } else if (this.lossWeights.size() > this.getLabels().size())
            throw new IllegalArgumentException("The number of loss weights cannot be more than the number of labels.");
        // Insure the number of filters or strides is not greater than the number of convolutions.
        if (! this.convolutions.isEmpty()) {
            if (this.filterSizes.size() > this.convolutions.size())
                throw new IllegalArgumentException("Cannot have more filters than convolution layers.");
            if (this.strides.size() > this.convolutions.size())
                throw new IllegalArgumentException("Cannot have more strides than convolution layers.");
        }
        // Verify that the sub-sampling factor is in range. This requires us to compute
        // the effect of the various input layers.  We need this for the
        // balanced-layer computation, too.
        LayerWidths widthComputer = new LayerWidths(this.reader.getWidth(),
                this.getChannelCount());
        if (! this.convolutions.isEmpty()) {
            int strideFactor = this.strides.first();
            int filters = this.filterSizes.first();
            // Compute the width after the last convolution.
            for (int cWidth : this.convolutions) {
                widthComputer.applyConvolution(cWidth, strideFactor, filters);
                strideFactor = this.strides.softNext();
                filters = this.filterSizes.softNext();
            }
            if (widthComputer.getOutWidth() < 1) {
                throw new IllegalArgumentException("Convolution stride is too big, output width less than 1.");
            } else if (widthComputer.getOutWidth() <= this.subFactor) {
                throw new IllegalArgumentException("Subsampling factor must be less than " +
                        widthComputer.getOutWidth() + " in this configuration.");
            } else if (this.subFactor > 1) {
                widthComputer.applySubsampling(this.subFactor);
            }
        }
        // Flatten any leftover channel depth.
        widthComputer.flatten();
        // If we have balanced layers, we need to compute the hidden layer widths
        // here.
        if (this.balancedLayers > 0) {
            // Compute the balanced layer widths.
            int[] widths = widthComputer.balancedLayers(this.balancedLayers, this.getLabels().size());
            this.denseLayers = new IntegerList(widths);
        }
        // Save the regularization configuration.
        this.regulizer = new Regularization(this.regMode, this.regFactor);
        // Correct the early stop value.
        if (this.earlyStop == 0) this.earlyStop = Integer.MAX_VALUE;
        // Correct the Nesterov learning rate for the weight updater.  The default here is 0.1, not 1e-3
        this.realLearningRate = this.learnRate;
        if (this.weightUpdateMethod == GradientUpdater.Type.NESTEROVS)
            this.realLearningRate *= 100;
        // Initialize the low-level computed parameters.
        initializeBaseModelParms();
        // If the user asked for a configuration file, write it here.
        if (this.parmFile != null) writeParms(this.parmFile);
    }

    /** Write all the parameters to a configuration file.
    *
    * @param outFile	file to be created for future use as a configuration file
    *
    * @throws IOException */
    public abstract void writeParms(File outFile) throws IOException;

    /**
     * Set the defaults peculiar to the subclass.
     */
    public abstract void setSubclassDefaults();

    /**
     * Write the model-related parameters to a configuration file.
     *
     * @param writer	output stream for the configuration file
     */
    protected void writeModelParms(PrintWriter writer) {
        String commentFlag = "";
        writeBaseModelParms(writer);
        writer.format("--iter %d\t# number of training iterations per batch%n", this.iterations);
        writer.format("--batchSize %d\t# size of each training batch%n", this.batchSize);
        String functions = Stream.of(Trainer.Type.values()).map(Trainer.Type::name).collect(Collectors.joining(", "));
        writer.format("## Valid training methods are %s.%n", functions);
        writer.format("--method %s\t# training set processing method%n", this.method.toString());
        writer.format("--earlyStop %d\t# early-stop useless-iteration limit%n", this.earlyStop);
        if (this.denseLayers.isEmpty()) {
            writer.format("# --widths 10\t# configure number and widths of hidden layers%n");
            writer.println("--balanced 2\t# number of hidden layers (overrides widths)");
        } else {
            writer.format("--widths %s\t# configure hidden layers%n", this.denseLayers.original());
            writer.println("# --balanced 2\t# number of hidden layers (overrides widths)");
        }
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
        functions = Stream.of(LossFunctionType.values()).map(LossFunctionType::name).collect(Collectors.joining(", "));
        writer.format("## Valid loss functions are %s.%n", functions);
        writer.format("--lossFun %s\t# loss function for scoring output%n", this.lossFunction.name());
        String weightString;
        if (this.lossWeights.isEmpty())
            weightString = "# --weights 1.0";
        else
            weightString = String.format("--weights %s",  this.lossWeights.original());
        writer.format("%s\t# weights (by label) for computing loss function%n", weightString);
        functions = Stream.of(WeightInit.values()).map(WeightInit::name).collect(Collectors.joining(", "));
        writer.format("## Valid starting weight initializations are %s.%n", functions);
        writer.format("--start %s\t# starting weight initialization method%n", this.weightInitMethod.name());
        functions = Stream.of(Activation.values()).map(Activation::name).collect(Collectors.joining(", "));
        writer.format("## Valid activation functions are %s.%n", functions);
        writer.format("--init %s\t# initial activation function%n", this.initActivationType.name());
        writer.format("--activation %s\t# hidden layer activation function%n", this.activationType.name());
        functions = Stream.of(GradientNormalization.values()).map(GradientNormalization::name).collect(Collectors.joining(", "));
        writer.format("## Valid gradient normalizations are %s.%n", functions);
        writer.format("--gradNorm %s\t# gradient normalization strategy%n", this.gradNorm.toString());
        commentFlag = (this.batchNormFlag ? "" : "# ");
        writer.format("%s--batch\t# use a batch normalization layer%n", commentFlag);
        commentFlag = (this.convolutions.isEmpty() ? "# " : "");
        String cnn = (this.convolutions.isEmpty() ? "3" : this.convolutions.original());
        writer.format("%s--cnn %s\t# convolution kernel sizes%n", commentFlag, cnn);
        writer.format("%s--filters %s\t# number of convolution filters to try%n", commentFlag, this.filterSizes.original());
        writer.format("%s--sub %d\t# subsampling factor%n", commentFlag, this.subFactor);
        writer.format("%s--strides %s\t# stride to use for convolution layer%n", commentFlag, this.strides.original());
        commentFlag = (this.lstmLayers == 0 ? "# " : "");
        writer.format("%s--lstm %d\t# number of long-short-term time series layers%n", commentFlag, this.lstmLayers);
        functions = Stream.of(GradientUpdater.Type.values()).map(GradientUpdater.Type::name).collect(Collectors.joining(", "));
        writer.format("## Valid updater methods are %s.%n", functions);
        writer.format("--updater %s\t# weight gradient updater method (uses learning rate)%n", this.weightUpdateMethod.name());
        writer.format("--bUpdater %s\t# bias gradient updater method (uses update rate)%n", this.biasUpdateMethod.name());
        commentFlag = (this.rawMode ? "" : "# ");
        writer.format("%s--raw\t# suppress input normalization%n", commentFlag);
    }

    /**
     * @return a string builder displaying the structure of the model
     *
     * @param runStats	the results of training the model
     */
    protected TextStringBuilder displayModel(RunStats runStats) throws IOException {
        TextStringBuilder parms = new TextStringBuilder();
        parms.appendNewLine();
        parms.appendln(
                        "Model file is %s%n" +
                        "=========================== Parameters ===========================%n" +
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
                        "     Loss function label weights are %s.%n" +
                        "     %s minutes to run %d %s (best was %d), with %d score bounces.%n" +
                        "     Final score is %g.  %d models saved.",
                       this.modelName.getCanonicalPath(),
                       this.iterations, this.batchSize, this.testSize, this.seed,
                       this.subFactor, this.method.toString(), this.regulizer,
                       this.gradNorm.toString(), this.biasUpdateMethod.toString(), this.biasRate,
                       this.weightInitMethod.toString(), this.weightUpdateMethod.toString(), this.learnRate,
                       this.initActivationType.toString(), this.activationType.toString(),
                       outActivation.toString(), this.lossFunction.toString(),
                       this.lossWeights.toString(), runStats.getDuration(), runStats.getEventCount(),
                       runStats.getEventsName(), runStats.getBestEvent(), runStats.getBounceCount(),
                       runStats.getBestScore(), runStats.getSaveCount());
        if (! this.convolutions.isEmpty()) {
            parms.appendln("     Convolution layers used with kernel sizes %s", this.convolutions);
            parms.appendln("     Convolutions used filter sizes %s and strides %s.",
                    this.filterSizes, this.strides);
        }
        if (this.batchNormFlag)
            parms.appendln("     Batch normalization applied.");
        if (this.denseLayers.isEmpty())
            parms.appendln("     No hidden layers used.");
        else
            parms.appendln("     Hidden layer configuration is %s.", this.denseLayers);
        if (this.rawMode)
            parms.appendln("     Data normalization is turned off.");
        if (this.isChannelMode())
            parms.appendln("     Input uses channel vectors.");
        return parms;
    }

    /**
     * @return a model built from the definition parameters
     */
    protected MultiLayerNetwork buildModel() {
        if (! this.rawMode) {
            // Here the model must be normalized.
            log.info("Normalizing data using testing set.");
            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(this.getTestingSet());
            normalizer.transform(this.getTestingSet());
            setNormalizer(normalizer);
        }
        // Now we build the model configuration.
        LayerWidths widthComputer = new LayerWidths(this.reader.getWidth(), this.getChannelCount());
        log.info("Building model configuration with input width {} and {} channels.",
                widthComputer.getInWidth(), widthComputer.getChannels());
        NeuralNetConfiguration.ListBuilder configuration = new NeuralNetConfiguration.Builder()
                .seed(this.seed)
                .activation(activationType)
                .weightInit(this.weightInitMethod)
                .biasUpdater(GradientUpdater.create(this.biasUpdateMethod, this.biasRate))
                .updater(GradientUpdater.create(this.weightUpdateMethod, this.realLearningRate))
                .gradientNormalization(this.gradNorm).list();
        // Compute the input type.
        this.inputShape = InputType.convolutional(1, widthComputer.getInWidth(),
                    widthComputer.getChannels());
        configuration.setInputType(getInputShape());
        // This flag will be set if an input layer has been created.  If there is none, we need to put
        // in a dense layer.
        boolean inputLayerCreated = false;
        // This tracks the current layer number.
        int layerIdx = 0;
        if (! this.convolutions.isEmpty()) {
            // Create the convolution layers.  For the first layer, the channel depth is the number
            // of channels and the output size is the first filter size.
            int convOut = this.filterSizes.first();
            int strideFactor = this.strides.first();
            for (int convKernel : this.convolutions) {
                log.info("Creating convolution layer with {} inputs.", widthComputer.getOutWidth());
                configuration.layer(layerIdx++, new ConvolutionLayer.Builder().activation(this.initActivationType)
                        .nIn(widthComputer.getChannels()).nOut(convOut).kernelSize(1, convKernel)
                        .stride(1, strideFactor).build());
                // Compute the shape of the this layer's output.
                widthComputer.applyConvolution(convKernel, strideFactor, convOut);
                // Set up for the next layer.
                convOut = this.filterSizes.softNext();
                strideFactor = this.strides.softNext();
            }
            if (subFactor > 1) {
                log.info("Creating subsampling layer with {} inputs.", widthComputer.getOutWidth());
                configuration.layer(layerIdx++, new SubsamplingLayer.Builder()
                        .kernelSize(1, this.subFactor)
                        .stride(1, this.subFactor).build());
                // Reduce the input size by the subsampling factor.
                widthComputer.applySubsampling(this.subFactor);
            }
            inputLayerCreated = true;
        }
        if (this.lstmLayers > 0) {
            // Here we have LSTM layers.  First, we convert the input.
            configuration.inputPreProcessor(layerIdx, new CnnToRnnSequencePreprocessor());
            // Create the layers.
            for (int i = 0; i < this.lstmLayers; i++) {
                log.info("Creating LSTM layer {}.", i + 1);
                // Configure the builder.
                LSTM.Builder builder = new LSTM.Builder().activation(this.initActivationType)
                        .nIn(widthComputer.getChannels()).nOut(widthComputer.getChannels());
                // Do the regularization.
                this.regulizer.apply(builder);
                // Add the layer.
                configuration.layer(layerIdx++, builder.build());
            }
            // Now convert the output.
            configuration.inputPreProcessor(layerIdx, new RnnSequenceToFeedForwardPreProcessor());
            inputLayerCreated = true;
        }
        if (! inputLayerCreated || this.batchNormFlag) {
            // We need a 2D shape, and no one is going to flatten it, so we need to set up a flattener.
            configuration.inputPreProcessor(layerIdx, new CnnToFeedForwardPreProcessor(1, widthComputer.getOutWidth(),
                    widthComputer.getChannels()));
            widthComputer.flatten();
            // Add batch normalization if desired.
            if (this.batchNormFlag) {
                log.info("Adding batch normalization layer.");
                int width = widthComputer.getOutWidth();
                configuration.layer(new BatchNormalization.Builder().nIn(width).nOut(width).build());
            }
        } else {
            // It will be automatically flattened, so we need to update the width computation.
            widthComputer.flatten();
        }
        // We have multi-dimensional input, so we must flatten the width for the hidden layers.
        // Compute the hidden layers.
        int outputCount = this.getLabels().size();
        for (int layerSize : this.denseLayers) {
            // The new layer goes in here.
            Layer newLayer;
            if (layerSize == 0) {
                // Here we have an ElementWiseMultiplicationLayer.
                ElementWiseMultiplicationLayer.Builder builder = new ElementWiseMultiplicationLayer.Builder()
                        .nIn(widthComputer.getOutWidth()).nOut(widthComputer.getOutWidth()).activation(Activation.IDENTITY);
                newLayer = builder.build();
                log.info("Creating element-wise mulitplication layer with width {}.", widthComputer.getOutWidth());
            } else {
                // Here we have a DenseLayer.
                log.info("Creating hidden layer with input width {} and {} outputs.", widthComputer.getOutWidth(), layerSize);
                DenseLayer.Builder builder = new DenseLayer.Builder().nIn(widthComputer.getOutWidth()).nOut(layerSize);
                // If this is the first dense layer, apply the initial activation function.
                if (! inputLayerCreated) {
                    builder.activation(initActivationType);
                    inputLayerCreated = true;
                }
                // Do the regularization.
                this.regulizer.apply(builder);
                // Build the layer.
                newLayer = builder.build();
                // Update the width.
                widthComputer.applyFeedForward(layerSize);
            }
            // Add the layer.
            configuration.layer(newLayer);
        }
        // Add the output layer.
        this.outActivation = this.getOutActivation();
        ILossFunction lossComputer = this.lossFunction.create(this.lossWeights.getValues(this.getLabels().size()));
        log.info("Creating output layer with input width {} and {} outputs.", widthComputer.getOutWidth(),
                outputCount);
        configuration.layer(new OutputLayer.Builder()
                        .activation(outActivation)
                        .lossFunction(lossComputer)
                        .nIn(widthComputer.getOutWidth()).nOut(outputCount).build());
        // Here we create the model itself.
        log.info("Creating model.");
        MultiLayerNetwork model = new MultiLayerNetwork(configuration.build());
        model.init();
        return model;
    }

    /**
     * @return the output layer activation function for this processor
     */
    protected abstract Activation getOutActivation();

    /**
     * Specify the meta-column information.
     *
     * @param cols		array of metadata column names; the first is the sample ID, the last is the label (if any)
     */
    public abstract void setMetaCols(String[] cols);

    /**
     * @return the input shape for this model
     */
    public InputType getInputShape() {
        return inputShape;
    }

    /**
     * @return the loss function type
     */
    protected LossFunctionType getLossFunction() {
        return lossFunction;
    }

    /**
     * If a model was created and we are in production mode, save it to the model directory.
     *
     * @throws IOException
     */
    @Override
    protected void saveModel() throws IOException {
        if (this.production)
            super.saveModel();
    }

    /**
     * Force saving of a model to the model directory.
     */
    public void saveModelForced() throws IOException {
        super.saveModel();
    }

    /**
     * Force saving of the model to a specific file.
     *
     * @param file		file in which to save the model
     */
    public void saveModelForced(File file) throws IOException {
        this.modelName = file;
        saveModelForced();
    }

    /**
     * Test a model against a full training file.
     *
     * @param model		model to test
     * @param file		training file with which to test the model
     * @param reporter	report object for writing output
     *
     * @return an estimate of the prediction error
     *
     * @throws IOException
     */
    public IPredictError testPredictions(MultiLayerNetwork model, List<String> trainingFile, IValidationReport reporter) throws IOException {
        // Get access to the input data.
        TabbedDataSetReader batches = this.openDataFile(trainingFile);
        batches.setNormalizer(this.getNormalizer());
        // Initialize the error predictor.
        IPredictError errorPredictor = this.initializePredictError(this.getLabels(), trainingFile.size() - 1);
        // Initialize the output report.
        reporter.startReport(this.getMetaList(), this.getLabels());
        // Loop through the data, making predictions.
        for (DataSet batch : batches) {
            INDArray output = model.output(batch.getFeatures());
            errorPredictor.accumulate(batch.getLabels(), output);
            reporter.reportOutput(batch.getExampleMetaData(String.class), batch.getLabels(), output);
        }
        // Finish predicting.
        errorPredictor.finish();
        reporter.finishReport(errorPredictor);
        // Close the report output stream.
        reporter.close();
        // Return the evaluation of the predictions.
        return errorPredictor;
    }

    /**
     * Configure the model for training.  This includes parsing the header of the training/testing file,
     * reading the testing set, and initializing the model parameters.
     *
     * @param	reader for the testing/training data
     *
     * @throws IOException
     */
    public void configureTraining(TabbedDataSetReader myReader) throws IOException {
        this.configureReading(myReader);
        this.initializeTraining();
    }

    /**
     * Configure this processor with the specified parameters.
     *
     * @param parms				configuration parameters
     * @param modelDirectory	directory containing the model
     *
     * @return TRUE if successful, FALSE if the parameters were invalid
     *
     * @throws IOException
     */
    public boolean setupParameters(Parms parms, File modelDirectory) throws IOException {
        String[] parmArray = new String[parms.size() + 1];
        setAllDefaults();
        // Process the parameters.
        List<String> parmValues = parms.get();
        parmValues.add(modelDirectory.toString());
        parmArray = parmValues.toArray(parmArray);
        boolean retVal = this.parseArgs(parmArray);
        if (retVal) {
            // Setup the training configuration.
            this.setupTraining();
        }
        return retVal;
    }

    /**
     * Set all the default parameters for this processor.
     */
    public void setAllDefaults() {
        this.setSubclassDefaults();
        this.setDefaults();
        this.setModelDefaults();
    }

    /**
     * Initialize the reader for reading training and testing data.
     *
     * @param myReader	incoming dataset reader
     *
     * @throws IOException
     */
    public abstract void configureReading(TabbedDataSetReader myReader) throws IOException;

    /**
     * Initialize a prediction error object for this trainer.
     *
     * @param labels	labels for this model
     * @param rows		number of data rows to be predicted
     *
     * @return an object that can be used to compute prediction error
     */
    protected abstract IPredictError initializePredictError(List<String> labels, int rows);

    /**
     * Open a dataset iterator for the specified training file.
     *
     * @param strings	list of strings containing a training/testing set
     *
     * @return an Iterable for dataset batches in the training file
     *
     * @throws IOException
     */
    protected abstract TabbedDataSetReader openDataFile(List<String> trainingFile) throws IOException;

    /**
     * Create the run statistics for this training process.
     */
    protected abstract RunStats createRunStats(MultiLayerNetwork model, Trainer trainer);

    /**
     * Compute the evaluation report for this training process.
     *
     * @param bestModel		best model from the training
     * @param output		buffer to which the report is to be appended
     * @param runStats		run statistics from the training
     */
    protected abstract void report(MultiLayerNetwork bestModel, TextStringBuilder output, RunStats runStats);

    @Override
    public void run() {
        try {
            // Create the model.
            MultiLayerNetwork model = buildModel();
            // Train the model.
            Trainer trainer = Trainer.create(this.method, this, log);
            RunStats runStats = this.createRunStats(model, trainer);
            this.trainModel(model, runStats, trainer, this.getProgressMonitor());
            // Verify that the trained model is valid.
            if (! this.checkModel(model))
                throw new IllegalArgumentException("Model has overflowed with current parameters.");
            // Save the result.
            this.saveModel();
            // Display the configuration.
            MultiLayerNetwork bestModel = runStats.getBestModel();
            TextStringBuilder parms = displayModel(runStats);
            if (runStats.getSaveCount() == 0) {
                parms.appendNewLine();
                parms.appendln(UNDERFLOW_ERROR);
                this.getProgressMonitor().showMessage(UNDERFLOW_ERROR);
                this.clearRating();
            } else {
                this.report(bestModel, parms, runStats);
            }
            // Add the summary.
            String summary = bestModel.summary(getInputShape());
            parms.appendln(summary);
            // Add the parameter dump.
            parms.append(this.dumpModel(bestModel));
            // Output the result.
            String report = parms.toString();
            log.info(report);
            RunLog.writeTrialReport(this.getTrialFile(), this.comment, report);
            this.resultReport = this.comment + report;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * @return a data set reader for the specified string list
     *
     * @param strings	string list to use for training and testing
     *
     * @throws IOException
     */
    public abstract TabbedDataSetReader openReader(List<String> strings) throws IOException;

    /**
     * @return a data set reader for the specified file
     *
     * @param inFile	file to use for training and testing
     *
     * @throws IOException
     */
    public abstract TabbedDataSetReader openReader(File inFile) throws IOException;

    /**
     * @return the result report
     */
    public String getResultReport() {
        return this.resultReport;
    }

    /**
     * @return a testing-set error calculator for this processor
     */
    public abstract TestValidationReport getTestReporter();

    /**
     * Run predictions and output to a specific reporter.
     *
     * @param reporter	validation reporter
     * @param inFile	input file
     *
     * @throws IOException
     */
    public void runPredictions(IValidationReport reporter, File inFile) throws IOException {
        String idCol = this.getIdCol();
        if (idCol != null)
            reporter.setupIdCol(this.modelDir, idCol, this.getMetaList(), null);
        // Get the input data.
        Shuffler<String> inputData = new Shuffler<String>(1000);
        try (LineReader inStream = new LineReader(inFile)) {
            inputData.addSequence(inStream);
        }
        log.info("{} input data lines.", inputData.size() - 1);
        testModelPredictions(reporter, inputData);
    }

    /**
     * Run predictions for the current model and output to a specific reporter.
     *
     * @param reporter		validation reporter
     * @param inputData		input strings
     *
     * @throws IOException
     */
    protected void testModelPredictions(IValidationReport reporter, Shuffler<String> inputData) throws IOException {
        // Read the model.
        MultiLayerNetwork model = this.readModel();
        // Perform the prediction test.
        this.testPredictions(model, inputData, reporter);
    }

    /**
     * Compute the size-dependent default parameters
     *
     * @param inputSize		number of rows in input training file
     * @param featureCols	number of input columns
     */
    public void setSizeParms(int inputSize, int featureCols) {
        int testSize = inputSize / 10;
        if (testSize < 1) testSize = 1;
        this.setTestSize(testSize);
    }

    /**
     * @return the list of headers available for use as meta-columns
     *
     * @param headers	full list of column headers
     * @param labels	list of labels for this model
     */
    public abstract List<String> computeAvailableHeaders(List<String> headers, Collection<String> labels);

    /**
     * @return the relevant label column names for this model
     */
    public abstract List<String> getLabelCols();

    /**
     * @return the appropriate type of distributed output stream for this model
     */
    public abstract DistributedOutputStream getDistributor();

    /**
     * @param lossFunction 	new loss function for computing the score
     */
    public void setLossFunction(LossFunctionType lossFunction) {
        this.lossFunction = lossFunction;
    }

    /**
     * @return the prediction error from the best model found during a validation or search
     *
     * @param mainFile			list of testing-set records
     * @param testErrorReport	reporting facility for validation
     *
     * @throws IOException
     */
    public IPredictError testBestPredictions(List<String> mainFile, IValidationReport testErrorReport) throws IOException {
        MultiLayerNetwork model = this.getBestModel();
        IPredictError retVal = this.testPredictions(model, mainFile, testErrorReport);
        return retVal;
    }


}
