/**
 *
 */
package org.theseed.dl4j.decision;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.utils.IDescribable;

/**
 * A random forest is a set of decision trees, each trained on a randomly-selected subset of the
 * full training set.  An entire forest predicts an outcome by voting
 * @author Bruce Parrello
 *
 */
public class RandomForest implements Serializable {

    // FIELDS
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(RandomForest.class);
    /** serialization version ID */
    private static final long serialVersionUID = -5802362692626598850L;
    /** random number generator */
    private static Random rand = new Random();
    /** hyperparameters */
    private transient Parms parms;
    /** trees in this forest */
    private List<DecisionTree> trees;
    /** number of labels for this tree */
    private final int nLabels;
    /** number of input features for this tree */
    private int nFeatures;
    /** randomizer for selecting training sets */
    private transient IRandomizer randomizer;
    /** split point finder */
    private transient SplitPointFinder finder;

    /**
     * type of randomization
     *
     * BALANCED-- random with replacement, equal numbers of each class
     * UNIQUE-- random without replacement
     * RANDOM-- random with replacement
     */
    public static enum Method implements IDescribable {
        BALANCED {
            @Override
            IRandomizer create() {
                return new BalancedRandomizer();
            }

            @Override
            public String getDescription() {
                return "Class-balanced example sets with replacement.";
            }
        }, UNIQUE {
            @Override
            IRandomizer create() {
                return new NonReplacingRandomizer();
            }

            @Override
            public String getDescription() {
                return "Random example sets without replacement.";
            }
        }, RANDOM {
            @Override
            IRandomizer create() {
                return new ReplacingRandomizer();
            }

            @Override
            public String getDescription() {
                return "Random example sets with replacement.";
            }
        };

        /**
         * @return a randomizer of the appropriate type
         */
        abstract IRandomizer create();


    }

    /**
     * Initialize the randomizer with a specified seed.
     *
     * @param seed	randomization seed to use
     */
    public static void setSeed(int seed) {
        rand = new Random(seed);
    }

    /**
     * This class represents the hyperparameters for the random forest.
     */
    public static class Parms {

        /** number of trees */
        private int nTrees;
        /** number of features to use at each node */
        private int nFeatures;
        /** minimum number of features for a leaf node */
        private int leafLimit;
        /** number of examples to use for each tree */
        private int nExamples;
        /** type of randomization */
        private Method method;
        /** maximum tree depth */
        private int maxDepth;


        /**
         * Construct hyperparameters with default values.
         */
        public Parms() {
            this.nTrees = 50;
            this.nFeatures = 10;
            this.leafLimit = 1;
            this.nExamples = 1000;
            this.method = Method.RANDOM;
            this.maxDepth = 50;
        }

        /**
         * Construct hyperparameters with reasonable values for a specified training set.
         *
         * @param dataset	training set to use for computing the parameters
         */
        public Parms(DataSet dataset) {
            this.setup(dataset.numExamples(), dataset.numInputs());
        }

        /**
         * Construct hyperparameters with reasonable values for a training set with the
         * specified characteristics.
         *
         * @param nExamples		number of input rows
         * @param nInputs		number of feature columns
         */
        public Parms(int nExamples, int nInputs) {
            this.setup(nExamples, nInputs);
        }

        /**
         * Initialize the hyperparameters with reasonable values for a training set with the
         * specified characteristics.
         *
         * @param nRows		number of input rows
         * @param nInputs	number of feature columns
         */
        private void setup(int nRows, int nInputs) {
            this.nTrees = 50;
            int min = nInputs * 4 / this.nTrees;
            int middle = (int) Math.sqrt(nInputs) + 1;
            int max = nInputs / 2;
            this.nFeatures = (middle < min ? min : middle);
            if (this.nFeatures > max) this.nFeatures = max;
            this.leafLimit = 1;
            this.nExamples = nRows / 5;
            this.method = Method.RANDOM;
            this.maxDepth = 2 * nInputs;
        }

        /**
         * @return the number of trees to build
         */
        public int getNumTrees() {
            return this.nTrees;
        }

        /**
         * Specify the number of trees to build.
         *
         * @param nTrees 	the number of trees to build
         */
        public Parms setNumTrees(int nTrees) {
            this.nTrees = nTrees;
            return this;
        }

        /**
         * @return the number of features to test at each choice node
         */
        public int getNumFeatures() {
            return this.nFeatures;
        }

        /**
         * Set the number of features to test at each choice node.
         *
         * @param nFeatures 	the number of features to set
         */
        public Parms setNumFeatures(int nFeatures) {
            this.nFeatures = nFeatures;
            return this;
        }

        /**
         * @return the number of examples that triggers formation of a leaf node
         */
        public int getLeafLimit() {
            return this.leafLimit;
        }

        /**
         * Specify the number of examples that triggers formation of a leaf node.
         *
         * @param leafLimit 	the leafLimit to set
         */
        public Parms setLeafLimit(int leafLimit) {
            this.leafLimit = leafLimit;
            return this;
        }

        /**
         * @return the number of examples to use for each tree
         */
        public int getNumExamples() {
            return this.nExamples;
        }

        /**
         * Specify the number of examples to use for each tree.
         *
         * @param nExamples the number of examples to use
         */
        public Parms setnExamples(int nExamples) {
            this.nExamples = nExamples;
            return this;
        }

        /**
         * Specify the type of randomizer.
         *
         * @param method	randomization method
         */
        public Parms setMethod(Method method) {
            this.method = method;
            return this;
        }

        /**
         * @return the randomizer to use for selecting training sets
         */
        public IRandomizer getRandomizer() {
            return this.method.create();
        }

        /**
         * @return the randomizing method
         */
        public Method getMethod() {
            return this.method;
        }

        /**
         * @return the maximum permissible tree depth
         */
        public int getMaxDepth() {
            return this.maxDepth;
        }

        /**
         * Specify the maximum permissible tree depth
         *
         * @param maxDepth 	the depth to set
         */
        public Parms setMaxDepth(int maxDepth) {
            this.maxDepth = maxDepth;
            return this;
        }
    }

    /**
     * Construct a forest based on the specified training set.
     *
     * @param dataset	training set to use
     * @param parms		hyper-parameters
     * @param finder	split point finder
     */
    public RandomForest(DataSet dataset, Parms parms, SplitPointFinder finder) {
        this.nLabels = dataset.numOutcomes();
        this.nFeatures = dataset.numInputs();
        this.parms = parms;
        this.finder = finder;
        this.randomizer = parms.getRandomizer();
        // Initialize the randomizer.
        this.randomizer.initializeData(this.nLabels, this.parms.getNumExamples(), dataset);
        // Create an array of randomizer seeds, two per tree.
        long[] seeds1 = rand.longs(this.parms.getNumTrees()).toArray();
        long[] seeds2 = rand.longs(this.parms.getNumTrees()).toArray();
        // Create the decision trees in the random forest.
        this.trees = IntStream.range(0, this.parms.getNumTrees()).parallel()
                .mapToObj(i -> this.buildTree(seeds1[i], seeds2[i]))
                .collect(Collectors.toList());
    }

    /**
     * Create a decision tree from a balanced random subset of the rows in this dataset.
     * Note that all the trees are built in parallel, so care has been taken not to modify
     * the incoming parameters.
     *
     * @param seed1			seed to use for data randomization
     * @param seed2			seed to use for feature randomization
     *
     * @return a decision tree for the sampled subset
     */
    private DecisionTree buildTree(long seed1, long seed2) {
        // Get the sampling to use for training this tree.
        DataSet sample = this.randomizer.getData(seed1);
        // Build the decision tree.
        DecisionTree retVal = new DecisionTree(sample, this.parms, seed2, this.finder);
        return retVal;
    }

    /**
     * Predict the classifications for a set of features.
     *
     * @param features		array of features to predict
     */
    public INDArray predict(INDArray features) {
        // Get an empty result matrix.
        INDArray retVal = Nd4j.zeros(features.rows(), this.nLabels);
        // Ask each tree to vote.
        for (DecisionTree tree : this.trees)
            tree.vote(features, retVal);
        return retVal;
    }

    /**
     * Compute the impact of each input on the classifications.
     */
    public INDArray computeImpact() {
        INDArray retVal = Nd4j.zeros(this.nFeatures);
        // Accumulate each tree's impact.
        for (DecisionTree tree : this.trees)
            tree.accumulateImpact(retVal);
        // Take the mean.
        retVal.divi(this.trees.size());
        return retVal;
    }

    /**
     * Save this model to the specified file.
     *
     * @param saveFile	output file to contain this random forest
     *
     * @throws IOException
     * @throws FileNotFoundException
     */
    public void save(File saveFile) throws FileNotFoundException, IOException {
        try (FileOutputStream fileStream = new FileOutputStream(saveFile)) {
            ObjectOutputStream outStream = new ObjectOutputStream(fileStream);
            outStream.writeObject(this);
        }
    }

    /**
     * @return a random forest model loaded from the specified file
     *
     * @param loadFile		file containing the model
     */
    public static RandomForest load(File loadFile) throws IOException {
        RandomForest retVal;
        try (FileInputStream fileStream = new FileInputStream(loadFile)) {
            ObjectInputStream inStream = new ObjectInputStream(fileStream);
            retVal = (RandomForest) inStream.readObject();
        } catch (ClassNotFoundException e) {
            throw new IOException("Invalid format for " + loadFile + ": " + e.getMessage());
        }
        return retVal;
    }
}
