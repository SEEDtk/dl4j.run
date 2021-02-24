/**
 *
 */
package org.theseed.dl4j.decision;

import java.io.Serializable;
import java.util.Random;
import java.util.stream.IntStream;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A decision tree is a serializable data structure that can be used to classify an item based on a set of
 * features.  Each node of the tree specifies a feature and a threshold.  Values less than or equal to the
 * threshold are classified on the left and those greater than the threshold are classified on the right.
 * At the leaf level an output class is specified.
 *
 * @author Bruce Parrello
 *
 */
public class DecisionTree implements Serializable {

    // FIELDS
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(DecisionTree.class);
    /** number of features */
    private int nFeatures;
    /** number of classes */
    private int nClasses;
    /** root node */
    private Node root;
    /** hyperparameters */
    private transient RandomForest.Parms parms;
    /** randomizer */
    private transient Random randomizer;
    /** log base 2 factor */
    private static double LOG2BASE = Math.log(2.0);
    /** object ID for serialization */
    private static final long serialVersionUID = 4184229432605504478L;

    /**
     * This nested class represents a tree node.
     */
    public static abstract class Node implements Serializable {

        /** serialization ID */
        private static final long serialVersionUID = 6974234637504879773L;
        /** entropy value at this node */
        private double entropy;

        protected Node(double entropy) {
            this.entropy = entropy;
        }

        /**
         * @return the entropy
         */
        public double getEntropy() {
            return this.entropy;
        }

        /**
         * Accumulate the impact of this node's feature on the result.
         *
         * @param vector	vector of feature impacts
         */
        protected abstract void addImpact(INDArray vector);

    }

    /**
     * This is a decision node, with two children.
     */
    public static class ChoiceNode extends Node {

        /** serialization ID */
        private static final long serialVersionUID = -3772567268579272613L;
        /** index of deciding feature */
        private int iFeature;
        /** threshold (inclusive on the left) */
        private double limit;
        /** impurity gain */
        private double gain;
        /** left child */
        private Node left;
        /** right child */
        private Node right;

        /**
         * Create a new node with the specified decision criteria.
         *
         * @param iFeat		index of deciding feature
         * @param lim		maximum feature value for left children
         * @param entropy	entropy of this node
         * @param gain		impurity gain
         */
        protected ChoiceNode (int iFeat, double lim, double entropy, double gain) {
            super(entropy);
            this.iFeature = iFeat;
            this.limit = lim;
            this.gain = gain;
            this.left = null;
            this.right = null;
        }

        /**
         * Return the child of this node relevant to the specified input row.
         *
         * @param features	array of features containing the input row
         * @param idx		row to test
         *
         * @return the child node for the selected feature
         */
        public Node choose(INDArray features, int idx) {
            double value = features.getDouble(idx, this.iFeature);
            Node retVal = (value > limit ? this.right : this.left);
            return retVal;
        }

        /**
         * Attach the left-side child.
         *
         * @param left 	the left-side child node
         */
        public void setLeft(Node left) {
            this.left = left;
        }

        /**
         * Attach the right-side child.
         *
         * @param right 	the right-side child node
         */
        public void setRight(Node right) {
            this.right = right;
        }

        /**
         * @return the impurity gain for this node
         */
        public double getGain() {
            return this.gain;
        }

        /**
         * @return the decision feature index for this node
         */
        public int getFeatureIdx() {
            return this.iFeature;
        }

        @Override
        protected void addImpact(INDArray vector) {
            vector.putScalar(this.iFeature, this.gain + vector.getDouble(this.iFeature));
            this.left.addImpact(vector);
            this.right.addImpact(vector);
        }

        @Override
        public String toString() {
            return "ChoiceNode@" + Integer.toHexString(System.identityHashCode(this)) +
                    "[iFeature=" + this.iFeature + ", limit=" + this.limit + "]";
        }

    }

    /**
     * This class represents a leaf node.
     */
    public static class LeafNode extends Node {

        // FIELDS
        /** index of predicted class */
        private int iClass;
        /** serialization type ID */
        private static final long serialVersionUID = 3328808945708275642L;

        /**
         * Construct a leaf node.
         *
         * @param iClass	index of class decided by leaf node
         * @param entropy	entropy of this node
         */
        protected LeafNode(int iClass, double entropy) {
            super(entropy);
            this.iClass = iClass;
        }

        /**
         * @return the class decided by this leaf
         */
        public int getiClass() {
            return this.iClass;
        }

        @Override
        protected void addImpact(INDArray vector) {
            // A leaf has no impact.
        }

        @Override
        public String toString() {
            return "LeafNode@" + Integer.toHexString(System.identityHashCode(this)) +
                    "[iClass=" + this.iClass + "]";
        }

    }

    /**
     * Create a decision tree for the specified dataset.
     *
     * @param dataset	dataset to use for training the tree
     * @param parms		hyperparameter specification
     */
    public DecisionTree(DataSet dataset, RandomForest.Parms parms, long randSeed) {
        this.nClasses = dataset.numOutcomes();
        this.nFeatures = dataset.numInputs();
        this.parms = parms;
        this.randomizer = new Random(randSeed);
        // Compute the number of features to use in each tree.
        int arraySize = parms.getNumFeatures();
        if (this.nFeatures < arraySize) arraySize = this.nFeatures;
        // Get the features to use.
        int[] features = this.featuresToUse(arraySize);
        // Compute the starting entropy.
        double entropy = DecisionTree.entropy(dataset);
        // Create the root node.
        this.root = this.computeNode(dataset, 0, features, entropy);
    }

    /**
     * Create an array of randomly-selected indices specifying the features to use.
     *
     * @param arraySize		number of features to select
     *
     * @return an array of the specified size using
     */
    private int[] featuresToUse(int arraySize) {
        // Get all the possible feature indices in an array.
        int[] range = IntStream.range(0, this.nFeatures).toArray();
        // Form the output array.
        int[] retVal = new int[arraySize];
        // Loop through the range array, picking random items.
        for (int i = 0; i < arraySize; i++) {
            int rand = this.randomizer.nextInt(this.nFeatures - i);
            retVal[i] = range[rand + i];
            range[rand + i] = range[i];
        }
        return retVal;
    }

    /**
     * @return the entropy value of the dataset
     *
     * @param dataset	dataset whose entropy is desired
     */
    public static double entropy(DataSet dataset) {
        double total = dataset.numExamples();
        double retVal = 0.0;
        if (total > 0) {
            INDArray labelCounts = getLabelSums(dataset);
            retVal = labelEntropy(labelCounts);
        }
        return retVal;
    }

    /**
     * @return the entropy indicated by an array of label counts
     *
     * @param labelSum	array of label counts
     */
    private static double labelEntropy(INDArray labelSum) {
        double total = labelSum.sumNumber().doubleValue();
        double retVal = 0.0;
        for (int i = 0; i < labelSum.columns(); i++) {
            double pI = labelSum.getDouble(i) / total;
            if (pI > 0.0)
                retVal -= pI * Math.log(pI);
        }
        retVal /= LOG2BASE;
        return retVal;
    }

    /**
     * Recursively compute the tree node from the specified list of features.
     *
     * @param dataset			dataset of rows to be classified by this node
     * @param depth				depth of the node in question
     * @param featuresToUse		array of feature indices to use
     * @param entropy			entropy of the set
     *
     * @return a node for deciding this set
     */
    private Node computeNode(DataSet dataset, int depth, int[] featuresToUse, double entropy) {
        Node retVal;
        // Is this a leaf?
        if (dataset.numExamples() <= this.parms.getLeafLimit() || entropy <= 0.0 || depth >= this.parms.getMaxDepth()) {
            // Yes.  Assign the best label value.
            retVal = createLeaf(dataset, entropy);
        } else {
            // Loop through the features, looking for the one that creates the greatest entropy decrease.
            INDArray features = dataset.getFeatures();
            INDArray labels = dataset.getLabels();
            // These variables contain the data we need to split the dataset for the children.
            double bestMean = 0.0;
            int bestFeature = -1;
            double bestLeftEntropy = 0.0;
            double bestRightEntropy = 0.0;
            int bestLeftCount = 0;
            int bestRightCount = 0;
            double bestGain = 0.0;
            // Loop through the features left to examine.
            for (int i : featuresToUse) {
                double mean = features.getColumn(i).meanNumber().doubleValue();
                INDArray leftLabels = Nd4j.zeros(this.nClasses);
                INDArray rightLabels = Nd4j.zeros(this.nClasses);
                for (int j = 0; j < features.rows(); j++) {
                    INDArray labelArray = labels.getRow(j);
                    if (features.getDouble(j, i) <= mean)
                        leftLabels.addi(labelArray);
                    else
                        rightLabels.addi(labelArray);
                }
                // Now compute the information gain.  Note we skip if one of the branches is empty.
                double leftCount = leftLabels.sumNumber().doubleValue();
                double rightCount = rightLabels.sumNumber().doubleValue();
                if (leftCount > 0.0 && rightCount > 0.0) {
                    double leftEntropy = labelEntropy(leftLabels);
                    double rightEntropy = labelEntropy(rightLabels);
                    double totalCount = leftCount + rightCount;
                    double gain = entropy - (leftEntropy * leftCount + rightEntropy * rightCount) / totalCount;
                    // If this is our biggest gain, save it.
                    if (gain > bestGain) {
                        bestMean = mean;
                        bestFeature = i;
                        bestLeftEntropy = leftEntropy;
                        bestRightEntropy = rightEntropy;
                        bestGain = gain;
                        bestLeftCount = (int) leftCount;
                        bestRightCount = (int) rightCount;
                    }
                }
            }
            // If we were not able to gain anything, this is a leaf.
            if (bestFeature < 0)
                retVal = this.createLeaf(dataset, entropy);
            else {
                // Here we can split the node.
                ChoiceNode newNode = new ChoiceNode(bestFeature, bestMean, entropy, bestGain);
                // Split the incoming dataset.
                INDArray leftFeatures = Nd4j.zeros(bestLeftCount, this.nFeatures);
                INDArray leftLabels = Nd4j.zeros(bestLeftCount, this.nClasses);
                INDArray rightFeatures = Nd4j.zeros(bestRightCount, this.nFeatures);
                INDArray rightLabels = Nd4j.zeros(bestRightCount, this.nClasses);
                int leftCount = 0;
                int rightCount = 0;
                for (int i = 0; i < dataset.numExamples(); i++) {
                    if (features.getDouble(i, bestFeature) <= bestMean)
                        putRow(leftFeatures, leftLabels, leftCount++, dataset, i);
                    else
                        putRow(rightFeatures, rightLabels, rightCount++, dataset, i);
                }
                DataSet leftDataset = new DataSet(leftFeatures, leftLabels);
                DataSet rightDataset = new DataSet(rightFeatures, rightLabels);
                // Create the left node.
                newNode.setLeft(this.computeNode(leftDataset, depth + 1, featuresToUse, bestLeftEntropy));
                // Create the right node.  Here we can reuse the old bitmap.
                newNode.setRight(this.computeNode(rightDataset, depth + 1, featuresToUse, bestRightEntropy));
                // Return the new node.
                retVal = newNode;
            }
        }
        return retVal;
    }

    /**
     * Store a row from the dataset into the feature and label arrays.
     *
     * @param features	output feature array
     * @param labels	output label array
     * @param i			output row index
     * @param dataset	input dataset
     * @param i2		input row index
     */
    protected static void putRow(INDArray features, INDArray labels, int i, DataSet dataset, int i2) {
        features.putRow(i, dataset.getFeatures().getRow(i2));
        labels.putRow(i, dataset.getLabels().getRow(i2));
    }

    /**
     * @return a leaf node for a dataset
     *
     * @param dataset	dataset represented by the leaf
     * @param entropy	entropy of the dataset
     */
    private Node createLeaf(DataSet dataset, double entropy) {
        int label = DecisionTree.bestLabel(dataset);
        return new LeafNode(label, entropy);
    }

    /**
     * @return the index of the most popular label in the dataset
     *
     * @param dataset	dataset to process
     */
    public static int bestLabel(DataSet dataset) {
        INDArray labelSums = getLabelSums(dataset);
        int retVal = 0;
        double max = labelSums.getDouble(0);
        for (int i = 1; i < labelSums.columns(); i++) {
            double val = labelSums.getDouble(i);
            if (val > max) {
                retVal = i;
                max = val;
            }
        }
        return retVal;
    }

    /**
     * @return a vector containing the number of occurrences of each label in the dataset
     *
     * @param dataset	dataset of interest
     */
    private static INDArray getLabelSums(DataSet dataset) {
        return dataset.getLabels().sum(0);
    }

    /**
     * @return the index of the predicted label for the specified data row
     *
     * @param features	array of input rows
     * @param idx		index of row whose label is desired
     */
    public int predict(INDArray features, int idx) {
        Node current = root;
        log.debug("Predicting row {} using tree {}.", idx, current);
        while (current instanceof ChoiceNode) {
            ChoiceNode curr = (ChoiceNode) current;
            current = curr.choose(features, idx);
            log.debug("Node {} chosen.", current);
        }
        LeafNode curr = (LeafNode) current;
        return curr.getiClass();
    }

    /**
     * Add this tree's vote to the predictions of a dataset.
     *
     * @param features	array of input rows
     * @param labels	array in which to put label votes
     */
    public void vote(INDArray features, INDArray labels) {
        for (int i = 0; i < features.rows(); i++) {
            // Compute the prediction for this row.
            int label = this.predict(features, i);
            // Add our vote to its label output.
            labels.putScalar(i, label, labels.getDouble(i, label) + 1.0);
        }
    }

    /**
     * @return an estimate of relative variable impact for each input variable
     */
    public INDArray computeImpact() {
        // Denote each variable initially has zero impact.
        INDArray retVal = Nd4j.zeros(this.nFeatures);
        accumulateImpact(retVal);
        // Return the result.
        return retVal;
    }

    /**
     * Add this tree's impact into an impact array.
     *
     * @param impactArray	array in which to accumulate this tree's impact
     */
    protected void accumulateImpact(INDArray impactArray) {
        // Process all the nodes.
        this.root.addImpact(impactArray);
    }

}
