package org.acme;

public class DjlMain {
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.out.println("Usage: java -jar djl-sharded.jar <model> <batchsize>");
            System.exit(1);
        }
        if (args[0].equals("vgg16")) {
            new DjlVGG16().start(args[1]);
        } else if (args[0].equals("resnet")) {
            new DjlResnet().start(args[1]);
        } else {
            System.out.println("Unknown model: " + args[0]);
            System.exit(1);
        }
    }
}
