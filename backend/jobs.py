import threading
import time
from uuid import uuid4


class JobStore:
    def __init__(self):
        self._jobs = {}
        self._lock = threading.Lock()

    def create_job(self, job_type, params=None):
        job_id = uuid4().hex
        now = time.time()
        job = {
            "id": job_id,
            "type": job_type,
            "status": "queued",
            "paused": False,
            "cancelled": False,
            "progress": 0.0,
            "message": "Queued",
            "params": params or {},
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "ended_at": None,
            "total_steps": None,
            "history": [],
            "result": None,
        }
        with self._lock:
            self._jobs[job_id] = job
        return job

    def start_job(self, job_id, total_steps=None):
        now = time.time()
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            if job.get("cancelled"):
                job["status"] = "cancelled"
                job["ended_at"] = now
                job["updated_at"] = now
                return job
            job["status"] = "running"
            job["paused"] = False
            job["message"] = "Running"
            job["started_at"] = now
            job["updated_at"] = now
            if total_steps is not None:
                job["total_steps"] = total_steps
            return job

    def update_job(self, job_id, **updates):
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            job.update(updates)
            job["updated_at"] = time.time()
            return job

    def append_history(self, job_id, entry):
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            job["history"].append(entry)
            job["updated_at"] = time.time()
            return job

    def finish_job(self, job_id, status, result=None, message=None):
        now = time.time()
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            job["status"] = status
            if status in {"cancelled", "failed"}:
                job["paused"] = False
            job["ended_at"] = now
            job["updated_at"] = now
            if message:
                job["message"] = message
            if result is not None:
                job["result"] = result
            job["progress"] = 1.0
            return job

    def pause_job(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            if job.get("status") not in {"running"}:
                return job
            job["paused"] = True
            job["status"] = "paused"
            job["message"] = "Paused"
            job["updated_at"] = time.time()
            return job

    def resume_job(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            if job.get("status") not in {"paused"}:
                return job
            if job.get("cancelled"):
                job["status"] = "cancelled"
                job["ended_at"] = time.time()
                job["updated_at"] = time.time()
                return job
            job["paused"] = False
            job["status"] = "running"
            job["message"] = "Running"
            job["updated_at"] = time.time()
            return job

    def cancel_job(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            job["cancelled"] = True
            if job.get("status") in {"queued", "running", "paused"}:
                job["status"] = "cancelling"
                job["message"] = "Cancellation requested"
            job["updated_at"] = time.time()
            return job

    def get_job(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            return dict(job)

    def list_jobs(self):
        with self._lock:
            return [dict(job) for job in self._jobs.values()]
