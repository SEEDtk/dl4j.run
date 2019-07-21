package org.theseed.dl4j.train;


/**
 * Hello world!
 *
 */
public class App
{
    public static void main( String[] args )
    {
        TrainingProcessor runObject = new TrainingProcessor();
        boolean ok = runObject.parseCommand(args);
        if (ok) {
            runObject.run();
        }
    }
}
