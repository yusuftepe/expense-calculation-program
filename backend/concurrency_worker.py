"""
Paralel ve Arka Plan Görev Yöneticisi (concurrency_worker.py).

Bu modül; COMP 303 dersinin 'Concurrency / Async' ve 'OOP' gereksinimlerini
karşılamak üzere ThreadPoolExecutor tabanlı bir görev kuyruğu ve paralel
toplu fiş tarama / veri işleme altyapısı sunar.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from uuid import uuid4

from backend.decorators import execution_timer, logger
from backend.services.receipt_service import scan_receipt

TaskStatus = str  # "pending" | "running" | "completed" | "failed"


@dataclass
class TaskRecord:
    """Arka planda çalışan bir görevin durumunu ve sonucunu temsil eden veri sınıfı."""

    task_id: str
    name: str
    status: TaskStatus = "pending"
    result: Any = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    duration_ms: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "duration_ms": self.duration_ms,
        }


class BackgroundTaskManager:
    """
    ThreadPoolExecutor kullanarak OCR taraması, toplu fiş işleme ve paralel
    analiz görevlerini ana Flask iş parçacığını (main thread) bloklamadan yönetir.
    """

    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max(1, int(max_workers))
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="ExpenseWorker",
        )
        self._tasks: dict[str, TaskRecord] = {}
        self._futures: dict[str, Future[Any]] = {}
        self._lock = threading.Lock()

    def submit_task(
        self,
        func: Callable[..., Any],
        *args: Any,
        task_name: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Bir fonksiyonu arka planda çalışmak üzere iş parçacığı havuzuna (ThreadPool)
        gönderir ve takip edilebilmesi için benzersiz bir 'task_id' döndürür.
        """
        task_id = uuid4().hex[:12]
        name = task_name or getattr(func, "__name__", "background_task")
        record = TaskRecord(task_id=task_id, name=name, status="pending")

        with self._lock:
            self._tasks[task_id] = record

        def _wrapped_runner() -> Any:
            start = time.perf_counter()
            with self._lock:
                self._tasks[task_id].status = "running"
            try:
                output = func(*args, **kwargs)
                elapsed = round((time.perf_counter() - start) * 1000.0, 2)
                with self._lock:
                    self._tasks[task_id].status = "completed"
                    self._tasks[task_id].result = output
                    self._tasks[task_id].duration_ms = elapsed
                return output
            except Exception as exc:
                elapsed = round((time.perf_counter() - start) * 1000.0, 2)
                logger.error("Arka plan görevi başarısız oldu (%s): %s", name, exc)
                with self._lock:
                    self._tasks[task_id].status = "failed"
                    self._tasks[task_id].error = str(exc)
                    self._tasks[task_id].duration_ms = elapsed
                raise

        future = self._executor.submit(_wrapped_runner)
        with self._lock:
            self._futures[task_id] = future

        return task_id

    def get_task_status(
        self,
        task_id: str,
        wait: bool = False,
        timeout: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Belirtilen 'task_id' için güncel görev durumunu döndürür.
        'wait=True' verilirse görevin tamamlanmasını bekler.
        """
        with self._lock:
            future = self._futures.get(task_id)

        if wait and future is not None:
            try:
                future.result(timeout=timeout)
            except Exception:
                # Hata bilgisi zaten _tasks[task_id].error içine kaydedildi
                pass

        with self._lock:
            record = self._tasks.get(task_id)
            return record.to_dict() if record else None

    @execution_timer
    def run_parallel_tasks(
        self,
        callables: list[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]],
        timeout: Optional[float] = 30.0,
    ) -> list[dict[str, Any]]:
        """
        Birden fazla bağımsız görevi aynı anda (paralel) çalıştırır ve
        tümünün sonuçlarını sıralı bir liste olarak döndürür.
        """
        futures_map: dict[Future[Any], int] = {}
        results: list[dict[str, Any]] = [
            {"status": "pending", "result": None, "error": None}
            for _ in range(len(callables))
        ]

        for idx, (fn, args, kwargs) in enumerate(callables):
            fut = self._executor.submit(fn, *args, **kwargs)
            futures_map[fut] = idx

        for completed_fut in as_completed(futures_map, timeout=timeout):
            idx = futures_map[completed_fut]
            try:
                res = completed_fut.result()
                results[idx] = {"status": "completed", "result": res, "error": None}
            except Exception as exc:
                results[idx] = {"status": "failed", "result": None, "error": str(exc)}

        return results

    def shutdown(self, wait: bool = True) -> None:
        """İş parçacığı havuzunu güvenli biçimde kapatır."""
        self._executor.shutdown(wait=wait)


# Uygulama genelinde kullanılacak varsayılan görev yöneticisi nesnesi
task_worker = BackgroundTaskManager(max_workers=4)


@execution_timer
def scan_receipts_batch(image_paths: list[str]) -> list[dict[str, Any]]:
    """
    Birden fazla fiş görselini ThreadPoolExecutor üzerinden eşzamanlı (concurrent)
    olarak tarar ve her birinin sonucunu liste halinde döndürür.
    """
    if not image_paths:
        return []

    tasks = [(scan_receipt, (path,), {}) for path in image_paths]
    return task_worker.run_parallel_tasks(tasks)
