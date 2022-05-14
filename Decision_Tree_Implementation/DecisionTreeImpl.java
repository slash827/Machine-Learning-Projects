package com.company;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Fill in the implementation details of the class DecisionTree using this file. Any methods or
 * secondary classes that you want are fine but we will only interact with those methods in the
 * DecisionTree framework.
 *
 * You must add code for the 1 member and 4 methods specified below.
 *
 * See DecisionTree for a description of default methods.
 */
public class DecisionTreeImpl extends DecisionTree {
    private DecTreeNode root;
    //ordered list of class labels
    private List<String> labels;
    //ordered list of attributes
    private List<String> attributes;
    //map to ordered discrete values taken by attributes
    private Map<String, List<String>> attributeValues;

    /**
     * Answers static questions about decision trees.
     */
    public DecisionTreeImpl() {
        // no code necessary this is void purposefully
    }

    /**
     * Build a decision tree given only a training set.
     *
     * @param train: the training set
     */
    public DecisionTreeImpl(DataSet train) {
        this.labels = train.labels;
        this.attributes = train.attributes;
        this.attributeValues = train.attributeValues;

        boolean[] usedAttributes = new boolean[attributes.size()];
        root = new DecTreeNode("", "", null, false);
        buildDecisionTree(root, usedAttributes, train.instances);
    }

    // takes instances and returns "G" if all of them labeled G or "B" if all of them labeled B or "no" otherwise
    private String isAbsolute(List<Instance> instances) {
        String value = instances.get(0).label;
        for (Instance instance : instances)
            if(!value.equals(instance.label))
                return "no";

        return value;
    }

    /** takes the instances list and return "no" if they represent different samples otherwise
     * if all of them are the same but with different labels then it returns "G" or "B", as the "majority vote" or
     * "G" as a default if there are equal amount of "G" and "B"
     */
    private String isAllSamplesEqual(List<Instance> instances) {
        int countG=0, countB=0;
        List<String> firstAttributes = instances.get(0).attributes;
        for(Instance instance : instances) {
            if(!firstAttributes.equals(instance.attributes))
                return "no";
            if(instance.label.equals("G"))
                countG++;
            else
                countB++;
        }
        return countG >= countB ? "G" : "B";
    }

    // this is the recursive method that the constructor uses in order to build the decision tree
    private void buildDecisionTree(DecTreeNode node, boolean[] usedAttributes, List<Instance> instances) {
        // in case their is a node without instances then we choose "G" label as default
        if(instances.size() == 0) {
            node.terminal = true;
            node.label = "G";
            return;
        }

        String absoluteValue = isAbsolute(instances);
        if(!absoluteValue.equals("no")) {
            node.terminal = true;
            node.label = absoluteValue;
            return;
        }

        String allTheSame = isAllSamplesEqual(instances);
        if(!allTheSame.equals("no")) {
            node.terminal = true;
            node.label = allTheSame;
            return;
        }

        String highestAttribute = highestAttributeInfoGain(usedAttributes, instances);
        int highestAttributeIndex = attributes.indexOf(highestAttribute);

        node.attribute = highestAttribute;
        List<String> highestAttributeValues = this.attributeValues.get(highestAttribute);

        // here we will remove the attribute from further use with it's children
        usedAttributes[highestAttributeIndex] = true;

        // create child node for each attribute value of the selected attribute
        for (String value : highestAttributeValues) {

            // add only instances that are set with the required value
            List<Instance> newInstances = new ArrayList<>();
            for (Instance instance : instances)
                if (instance.attributes.get(highestAttributeIndex).equals(value))
                    newInstances.add(instance);

            DecTreeNode child = new DecTreeNode("", "", value, false);
            node.addChild(child);
            buildDecisionTree(child, usedAttributes, newInstances);
        }

        // here we will add the attribute for future use
        usedAttributes[highestAttributeIndex] = false;
    }

    // takes instances and attributes and return the attribute with the highest info gain
    private String highestAttributeInfoGain(boolean[] usedAttributes, List<Instance> instances) {
        double infoGain, maxInfoGain=0;
        String highestAttribute="";

        for(String attribute : attributes) {
            if(usedAttributes[attributes.indexOf(attribute)])
                continue;

            infoGain = attributeInfoGain(instances, attribute);
            if(infoGain > maxInfoGain) {
                maxInfoGain = infoGain;
                highestAttribute = attribute;
            }
        }
        return highestAttribute;
    }

    /**
     *
     * @param train is a dataset that we calculate it's infoGain
     */
    @Override
    public void rootInfoGain(DataSet train) {
        this.labels = train.labels;
        this.attributes = train.attributes;
        this.attributeValues = train.attributeValues;

        rootInfoGain(train.instances);
    }

    /**
     * Helper function
     */
    private void rootInfoGain(List<Instance> instances) {
        for(String attribute : attributes) {
            double infoGain = attributeInfoGain(instances, attribute);
            System.out.format("attribute name is: %s and info gain is: %.5f\n", attribute, infoGain);
        }
    }

    /**
     * Helper function
     */
    private double attributeInfoGain(List<Instance> instances, String attribute) {
        double infoGain = calcEntropyOfDataset(instances);
        int attributeIndex = getAttributeIndex(attribute);

        if(attributeIndex == -1)
            return 0;

        for (String value : attributeValues.get(attribute)) {
            double partialEntropy = calcValueEntropy(instances, attributeIndex, value);
            infoGain -= partialEntropy;
        }
        return infoGain;
    }

    /**
     * Helper function
     */
    private double calcValueEntropy(List<Instance> instances, int attributeIndex, String value) {
        int totalSamples=instances.size(), valuedSamples=0;
        List<Instance> valuedInstances = new ArrayList<>();

        for(Instance instance : instances)
            if(instance.attributes.get(attributeIndex).equals(value)) {
                valuedSamples++;
                valuedInstances.add(instance);
            }

        if(valuedSamples == 0)
            return 0;

        double probability = (double)valuedSamples / totalSamples;
        double entropy = calcEntropyOfDataset(valuedInstances);
        return probability * entropy;
    }

    /**
     * Helper function
     */
    private String classify(Instance instance, DecTreeNode node) {
        // in case it's a decision
        if(node.terminal)
            return node.label;

        int attributeIndex = getAttributeIndex(node.attribute);
        String instanceAttributeValue = instance.attributes.get(attributeIndex);
        int attributeValueIndex = getAttributeValueIndex(node.attribute, instanceAttributeValue);

        return classify(instance, node.children.get(attributeValueIndex));
    }

    /**
     *
     * @param instance takes one instance from the DataSet and classifies it
     * @return a String that represents the label of the classification result
     */
    @Override
    public String classify(Instance instance) {
        return classify(instance, this.root);
    }

    /**
     * Helper function
     */
    private double log2(double num) {
        return Math.log(num) / Math.log(2);
    }

    /**
     * Helper function
     */
    private double calcEntropyOfDataset(List<Instance> instances) {
        if(instances.size() == 0)
            return 0;

        double entropy=0;
        int goodAmount=0, badAmount=0;

        for(Instance instance : instances)
            if(instance.label.equals("G"))
                goodAmount++;
            else
                badAmount++;

        double factor = (double)goodAmount / instances.size();
        if(factor == 0)
            entropy = 0;
        else
            entropy -= factor * log2(factor);

        factor = (double)badAmount / instances.size();
        if(factor == 0)
            entropy = 0;
        else
            entropy -= factor * log2(factor);

        return entropy;
    }

    @Override
    public void printAccuracy(DataSet test) {
        int counter = 0;
        for(Instance testSample : test.instances) {
            String sampleLabel = this.classify(testSample);
            String actualLabel = testSample.label;
            if(actualLabel.equals(sampleLabel))
                counter++;
        }

        double accuracy = (double)counter / test.instances.size();
        System.out.format("%.5f\n", accuracy);
    }

    /**
     * Build a decision tree given a training set then prune it using a tuning set.
     * ONLY for extra credits
     * @param train: the training set
     * @param tune: the tuning set
     */
    DecisionTreeImpl(DataSet train, DataSet tune) {
        this(train);
        this.labels = train.labels;
        this.attributes = train.attributes;
        this.attributeValues = train.attributeValues;
    }

    /**
     * Print the decision tree in the specified format
     */
    @Override
    public void print() {
        printTreeNode(root, null, 0);
    }

    /**
     * Prints the subtree of the node with each line prefixed by 4 * k spaces.
     */
    public void printTreeNode(DecTreeNode p, DecTreeNode parent, int k) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < k; i++)
            sb.append("    ");

        String value;
        if (parent == null)
            value = "ROOT";
        else {
            int attributeValueIndex = this.getAttributeValueIndex(parent.attribute, p.parentAttributeValue);
            value = attributeValues.get(parent.attribute).get(attributeValueIndex);
        }
        sb.append(value);
        if (p.terminal) {
            sb.append(" (" + p.label + ")");
            System.out.println(sb.toString());
        } else {
            sb.append(" {" + p.attribute + "?}");
            System.out.println(sb.toString());
            for (DecTreeNode child : p.children)
                printTreeNode(child, p, k + 1);
        }
    }

    /**
     * Helper function to get the index of the label in labels list
     */
    private int getLabelIndex(String label) {
        for (int i = 0; i < this.labels.size(); i++)
            if (label.equals(this.labels.get(i)))
                return i;
        return -1;
    }

    /**
     * Helper function to get the index of the attribute in attributes list
     */
    private int getAttributeIndex(String attr) {
        for (int i = 0; i < this.attributes.size(); i++)
            if (attr.equals(this.attributes.get(i)))
                return i;
        return -1;
    }

    /**
     * Helper function to get the index of the attributeValue in the list for the attribute key in the attributeValues map
     */
    private int getAttributeValueIndex(String attr, String value) {
        for (int i = 0; i < attributeValues.get(attr).size(); i++)
            if (value.equals(attributeValues.get(attr).get(i)))
                return i;
        return -1;
    }
}
