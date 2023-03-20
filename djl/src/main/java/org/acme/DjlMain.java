package org.acme;

public class DjlMain {
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.out.println("Usage: java -jar djl-sharded.jar [learn/do]]");
            System.exit(1);
        }
        if (args[0].equals("learn")) {
            Learning.main(args);
        } else if (args[0].equals("do")) {
            Doing.main(args);
        } else {
            System.out.println("Unknown action: " + args[0]);
            System.exit(1);
        }
    }
}
