Maman 18
Author: Gilad Battat

The main function is in the file HW3.java it's execution should be with command line arguments which are files that contains written datasets as in the examples.
</br>
I've written the documentation of my code inside the file DecisionTreeImpl.java.
I've used a lot of private "Helper functions" and overloading to make the code more readable.
each function contains comments that explain each step, but most of them are clear anyway.

explanations:
rootInfoGain - i've used an overload that takes only the instances themselves.
then I've iterated with for loop on each of the attributes to calculate how much info gain does it
has on the dataset with a function called attributeInfoGain.
    attributeInfoGain - it uses another  two functions calcEntropyOfDataset which is obvious
                        and calcValueEntropy that takes a specific value of the attribute and checks
                        how much it separates the data well (just like the Decision tree algorithm in chapter 18).
then rootInfoGain prints the info of each attribute.

classify - i've used overload that takes also a specific node in the decision tree for which it
           will perform the calculation of which child node to go to (or make a decision and exit).

printAccuracy - runs the function classify for each data in the test and then calculates how many samples
                were labeled correctly divided by the total amount of samples = ( True Positive + True Negative ) / (all samples)

DecisionTreeImpl - creates a root node, a boolean array that stores the attributes that are currently in use so they
                   wouldn't be taken into account when we go down in the tree and calls buildDecisionTree to build the tree further.
                   <br>
                   buildDecisionTree - a pretty big function that build a specific node in the tree and checks the following:
                           If there are currently no instances in the dataset (because each split divides
                           the dataset by some proportion) then it will be a leaf with default value as "G".

                           If all the samples have the same label then it will be a leaf with that label.

                           If all the samples have the same attribute values exactly then we need that node to be a
                           leaf with a label this is the "majority vote".

                           the function also calls highestAttributeInfoGain that is self explanatory.
                           then the current node's attribute is that highest attribute and we mark this attribute as
                           "used" in the the usedAttributes array so that it's children couldn't use it.
                           then we create this node's children based on the amount of different values that our attribute can have.
                           each children has only a subset of it's father's instances and it is called recursively.
