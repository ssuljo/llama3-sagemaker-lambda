import boto3
import os
import json

# Initialize the boto3 client for SageMaker runtime
client = boto3.client(service_name="sagemaker-runtime")

# Load environment variables
MODEL_ID = os.environ.get("HF_MODEL_ID")
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME")

# Define parameters for LLM
parameters = {
    "model": MODEL_ID,
    "max_tokens": 3000,
    "temperature": 0.5,
    "top_p": 0.2,
}


def invoke_endpoint_for_inference(client, prompt, parameters):
    """
    Invoke the SageMaker endpoint with the given prompt and parameters.
    Returns the generated text from the model's response.
    """
    payload = {
        "messages": prompt,
        **parameters,
    }
    try:
        response = client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(payload),
        )
        response_body = response["Body"].read().decode("utf-8")
        response_json = json.loads(response_body)
        generated_text = response_json["choices"][0]["message"]["content"]
        return generated_text
    except Exception as e:
        print(f"Error invoking endpoint: {e}")
        raise


def validate_event(event):
    """
    Validate the incoming event to ensure it contains the required 'messages' key.
    """
    if "messages" not in event:
        raise ValueError("Missing 'messages' in the event payload.")
    return event["messages"]


def lambda_handler(event, context):
    """
    Main Lambda function handler.
    Processes the event, transforms the input data, invokes the SageMaker endpoint, and returns the response.
    """
    try:
        print("Received event: " + json.dumps(event, indent=2))

        messages = validate_event(event)

        response = invoke_endpoint_for_inference(client, messages, parameters)

        print(response)

        return {
            "statusCode": 200,
            "isBase64Encoded": False,
            "body": json.dumps(response),
        }

    except Exception as err:
        return {
            "statusCode": 400,
            "isBase64Encoded": False,
            "body": "Call Failed {0}".format(err),
        }
