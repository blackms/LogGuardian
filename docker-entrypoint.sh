#!/bin/bash
set -e

# Define function to display usage information
function show_usage {
    echo "Usage: $0 [COMMAND]"
    echo "Commands:"
    echo "  api              - Start the LogGuardian API server"
    echo "  train            - Run a training job"
    echo "  inference        - Run inference on a batch of logs"
    echo "  evaluate         - Run evaluation on test datasets"
    echo "  benchmark        - Run benchmarks"
    echo "  shell            - Start a shell session"
    echo "  python [args]    - Run a Python command"
    echo "  help             - Show this help message"
}

# If no arguments or help is requested, show usage
if [ "$1" = "" ] || [ "$1" = "help" ]; then
    show_usage
    exit 0
fi

# Handle different commands
case "$1" in
    api)
        # Start the API server
        echo "Starting LogGuardian API server..."
        python -m logguardian.api.server
        ;;
    train)
        # Run training job
        echo "Starting training job..."
        python -m logguardian.examples.train_with_three_stage "$@"
        ;;
    inference)
        # Run inference on a batch of logs
        echo "Running inference..."
        python -m logguardian.examples.inference "$@"
        ;;
    evaluate)
        # Run evaluation
        echo "Running evaluation..."
        python -m logguardian.examples.evaluate "$@"
        ;;
    benchmark)
        # Run benchmarks
        echo "Running benchmarks..."
        python -m logguardian.examples.benchmark_example "$@"
        ;;
    shell)
        # Start a shell
        echo "Starting shell..."
        exec /bin/bash
        ;;
    python)
        # Run python with remaining arguments
        shift
        exec python "$@"
        ;;
    *)
        # Unknown command
        echo "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac