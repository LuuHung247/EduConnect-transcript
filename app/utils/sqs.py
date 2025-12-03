import boto3
import json
import os
import logging

logger = logging.getLogger(__name__)

AWS_REGION = os.environ.get('AWS_REGION')
SQS_QUEUE_URL = os.environ.get('AWS_SQS_QUEUE_URL')

_sqs_client = None


def get_sqs_client():
    global _sqs_client
    if _sqs_client is None:
        _sqs_client = boto3.client('sqs', region_name=AWS_REGION)
    return _sqs_client


def send_transcript_job(job_data: dict) -> bool:
    """Send transcript job to SQS queue"""
    try:
        if not SQS_QUEUE_URL:
            logger.warning("⚠️ SQS_QUEUE_URL not configured")
            return False
        
        sqs = get_sqs_client()
        response = sqs.send_message(
            QueueUrl=SQS_QUEUE_URL,
            MessageBody=json.dumps(job_data),
            MessageAttributes={
                'JobType': {
                    'DataType': 'String',
                    'StringValue': 'transcript'
                }
            }
        )
        
        logger.info(f"📤 Job sent to SQS: {response.get('MessageId')}")
        return True
    
    except Exception as e:
        logger.error(f"❌ Failed to send SQS message: {e}")
        return False


def receive_transcript_job(wait_time: int = 20) -> dict:
    """Receive one job from SQS queue (long polling)"""
    try:
        sqs = get_sqs_client()
        response = sqs.receive_message(
            QueueUrl=SQS_QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=wait_time,
            MessageAttributeNames=['All']
        )
        
        messages = response.get('Messages', [])
        if not messages:
            return None
        
        message = messages[0]
        return {
            'receipt_handle': message['ReceiptHandle'],
            'body': json.loads(message['Body']),
            'message_id': message['MessageId']
        }
    
    except Exception as e:
        logger.error(f"❌ Failed to receive SQS message: {e}")
        return None


def delete_message(receipt_handle: str) -> bool:
    """Delete processed message from queue"""
    try:
        sqs = get_sqs_client()
        sqs.delete_message(
            QueueUrl=SQS_QUEUE_URL,
            ReceiptHandle=receipt_handle
        )
        logger.info("🗑️ Message deleted from queue")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to delete SQS message: {e}")
        return False