"""
IPD-QMA REST API

FastAPI-based REST API for IPD-QMA analysis.
Provides endpoints for:
- Asynchronous job processing
- Analysis submission
- Results retrieval
- Health checks
- Swagger documentation

Run with:
    uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

Or using Docker:
    docker build -t ipd-qma-api .
    docker run -p 8000:8000 ipd-qma-api
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import uuid
import asyncio
import os
import json
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipd_qma import IPDQMA, IQMAConfig
from ipd_qma_advanced import IPDQMAAdvanced
from ipd_qma_validation import IPDQMAValidator

# Create FastAPI app
app = FastAPI(
    title="IPD-QMA API",
    description="REST API for Individual Participant Data Quantile Meta-Analysis",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ========================================================================
# DATA MODELS
# ========================================================================

class StudyData(BaseModel):
    """Study data model."""
    control: List[float] = Field(..., description="Control group outcomes")
    treatment: List[float] = Field(..., description="Treatment group outcomes")
    name: Optional[str] = Field(None, description="Study name")


class AnalysisConfig(BaseModel):
    """Analysis configuration model."""
    quantiles: Optional[List[float]] = Field(None, description="Quantiles to analyze (0-1)")
    n_bootstrap: int = Field(500, description="Number of bootstrap samples", ge=100, le=10000)
    confidence_level: float = Field(0.95, description="Confidence level", ge=0.8, le=0.999)
    use_random_effects: bool = Field(True, description="Use random-effects model")
    tau2_estimator: str = Field("dl", description="Heterogeneity estimator", pattern="^(dl|pm)$")
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    n_workers: Optional[int] = Field(None, description="Number of parallel workers")
    show_progress: bool = Field(False, description="Show progress (not applicable in API)")


class AnalysisRequest(BaseModel):
    """Analysis request model."""
    studies: List[StudyData] = Field(..., description="Study data", min_items=1)
    config: AnalysisConfig = Field(default_factory=AnalysisConfig)
    validate_data: bool = Field(True, description="Validate data before analysis")


class JobStatus(BaseModel):
    """Job status model."""
    job_id: str
    status: str  # pending, running, completed, failed
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    progress: float = 0.0


class AnalysisResponse(BaseModel):
    """Analysis response model."""
    job_id: str
    status: str
    message: str
    results: Optional[Dict[str, Any]] = None


# ========================================================================
# IN-MEMORY JOB STORAGE
# ========================================================================

jobs = {}


# ========================================================================
# UTILITY FUNCTIONS
# ========================================================================

def generate_job_id() -> str:
    """Generate unique job ID."""
    return str(uuid.uuid4())


def process_analysis_job(job_id: str, request: AnalysisRequest):
    """Process analysis job asynchronously."""
    try:
        jobs[job_id].update({
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "progress": 0.1
        })

        # Validate data
        if request.validate_data:
            validator = IPDQMAValidator(strict=False)
            validation_results = validator.validate_studies([
                (np.array(s.control), np.array(s.treatment))
                for s in request.studies
            ])

            if not validation_results.get('overall', {}).get('passed', True):
                jobs[job_id].update({
                    "status": "failed",
                    "error": "Data validation failed",
                    "completed_at": datetime.now().isoformat()
                })
                return

        # Prepare configuration
        config = IQMAConfig(
            quantiles=request.config.quantiles,
            n_bootstrap=request.config.n_bootstrap,
            confidence_level=request.config.confidence_level,
            use_random_effects=request.config.use_random_effects,
            tau2_estimator=request.config.tau2_estimator,
            random_seed=request.config.random_seed,
            n_workers=request.config.n_workers,
            show_progress=False  # No progress bars in API
        )

        jobs[job_id].update({"progress": 0.3})

        # Prepare studies data
        studies_data = [
            (np.array(s.control), np.array(s.treatment))
            for s in request.studies
        ]

        # Run analysis
        analyzer = IPDQMA(config)
        results = analyzer.fit(studies_data)

        jobs[job_id].update({"progress": 0.8})

        # Convert results to JSON-serializable format
        results_json = {
            "n_studies": results["n_studies"],
            "model_type": results["model_type"],
            "profile": results["profile"].to_dict("records"),
            "slope_test": results["slope_test"],
            "lnvr_test": results["lnvr_test"],
            "config": {
                "quantiles": config.quantiles,
                "n_bootstrap": config.n_bootstrap,
                "confidence_level": config.confidence_level,
                "use_random_effects": config.use_random_effects
            }
        }

        jobs[job_id].update({
            "status": "completed",
            "results": results_json,
            "analyzer": analyzer,
            "progress": 1.0,
            "completed_at": datetime.now().isoformat()
        })

    except Exception as e:
        jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })


# ========================================================================
# API ENDPOINTS
# ========================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "IPD-QMA API",
        "version": "2.0.0",
        "description": "REST API for Individual Participant Data Quantile Meta-Analysis",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "analyze": "/api/v1/analyze",
            "jobs": "/api/v1/jobs/{job_id}"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }


@app.post("/api/v1/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def submit_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit an IPD-QMA analysis job.

    The analysis runs asynchronously. Use the returned job_id to check status
    and retrieve results.
    """
    # Validate request
    if not request.studies:
        raise HTTPException(status_code=400, detail="At least one study is required")

    # Create job
    job_id = generate_job_id()
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "request": request.dict(),
        "progress": 0.0
    }

    # Start background task
    background_tasks.add_task(process_analysis_job, job_id, request)

    return AnalysisResponse(
        job_id=job_id,
        status="pending",
        message="Analysis job submitted successfully"
    )


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatus, tags=["Jobs"])
async def get_job_status(job_id: str):
    """Get the status of an analysis job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        error=job.get("error"),
        progress=job.get("progress", 0.0)
    )


@app.get("/api/v1/jobs/{job_id}/results", tags=["Jobs"])
async def get_job_results(job_id: str):
    """
    Get the results of a completed analysis job.

    Returns the full analysis results including:
    - Quantile profile
    - Slope and lnVR tests
    - Study details
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job["status"] == "pending":
        raise HTTPException(status_code=202, detail="Job is pending")
    elif job["status"] == "running":
        raise HTTPException(status_code=202, detail="Job is running")
    elif job["status"] == "failed":
        raise HTTPException(
            status_code=500,
            detail=f"Job failed: {job.get('error', 'Unknown error')}"
        )

    # Return results
    return JSONResponse(content=job["results"])


@app.delete("/api/v1/jobs/{job_id}", tags=["Jobs"])
async def delete_job(job_id: str):
    """Delete a job and its results."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    del jobs[job_id]
    return {"message": f"Job {job_id} deleted"}


@app.get("/api/v1/jobs", tags=["Jobs"])
async def list_jobs(
    status: Optional[str] = None,
    limit: int = 100
):
    """List all jobs (with optional filtering)."""
    job_list = []

    for job_id, job in jobs.items():
        if status is None or job["status"] == status:
            job_list.append({
                "job_id": job_id,
                "status": job["status"],
                "created_at": job["created_at"],
                "progress": job.get("progress", 0.0)
            })

    # Sort by creation time (newest first)
    job_list.sort(key=lambda x: x["created_at"], reverse=True)

    return {"jobs": job_list[:limit]}


# ========================================================================
# ADVANCED ANALYSIS ENDPOINTS
# ========================================================================

@app.post("/api/v1/analyze/advanced", tags=["Analysis"])
async def submit_advanced_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit an advanced IPD-QMA analysis with additional features.

    Includes publication bias, subgroup analysis, and sensitivity analysis.
    """
    # Similar to regular analysis but uses IPDQMAAdvanced
    job_id = generate_job_id()
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "request": request.dict(),
        "advanced": True,
        "progress": 0.0
    }

    # Advanced processing function would go here
    # For now, use regular processing
    background_tasks.add_task(process_analysis_job, job_id, request)

    return AnalysisResponse(
        job_id=job_id,
        status="pending",
        message="Advanced analysis job submitted (currently using standard analysis)"
    )


# ========================================================================
# FILE UPLOAD ENDPOINTS
# ========================================================================

@app.post("/api/v1/upload", tags=["Upload"])
async def upload_file(
    file: UploadFile = File(...),
    study_id_col: str = "study_id",
    group_col: str = "group",
    outcome_col: str = "outcome"
):
    """
    Upload a CSV or Excel file with study data.

    Expected format:
    - study_id: Study identifier
    - group: 'control' or 'treatment'
    - outcome: Continuous outcome value

    Returns the parsed study data ready for analysis.
    """
    # Read file
    try:
        contents = await file.read()

        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Use CSV or Excel."
            )

        # Validate columns
        required_cols = [study_id_col, group_col, outcome_col]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing}"
            )

        # Parse into study format
        studies = []
        for study_id in df[study_id_col].unique():
            control = df[
                (df[study_id_col] == study_id) &
                (df[group_col] == 'control')
            ][outcome_col].values.tolist()

            treatment = df[
                (df[study_id_col] == study_id) &
                (df[group_col] == 'treatment')
            ][outcome_col].values.tolist()

            if len(control) > 0 and len(treatment) > 0:
                studies.append(StudyData(
                    control=control,
                    treatment=treatment,
                    name=f"Study_{study_id}"
                ))

        return {
            "filename": file.filename,
            "n_studies": len(studies),
            "studies": studies
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# VALIDATION ENDPOINTS
# ========================================================================

@app.post("/api/v1/validate", tags=["Validation"])
async def validate_data(studies: List[StudyData]):
    """
    Validate study data without running analysis.

    Returns:
    - Quality scores
    - Warnings
    - Errors
    - Suggestions
    """
    validator = IPDQMAValidator(strict=False)

    results = []
    for i, study in enumerate(studies):
        result = validator.validate_study(
            np.array(study.control),
            np.array(study.treatment),
            study.name or f"Study_{i+1}"
        )

        results.append({
            "study": study.name or f"Study_{i+1}",
            "passed": result.passed,
            "score": result.score,
            "warnings": result.warnings,
            "errors": result.errors
        })

    # Calculate overall score
    scores = [r["score"] for r in results]
    overall_score = np.mean(scores) if scores else 0

    return {
        "overall_passed": all(r["passed"] for r in results),
        "overall_score": overall_score,
        "studies": results
    }


# ========================================================================
# EXPORT ENDPOINTS
# ========================================================================

@app.get("/api/v1/jobs/{job_id}/export/csv", tags=["Export"])
async def export_csv(job_id: str):
    """Export job results as CSV file."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot export: job status is {job['status']}"
        )

    # Create DataFrame from profile
    df = pd.DataFrame(job["results"]["profile"])

    # Save to temp file
    temp_file = f"/tmp/{job_id}_results.csv"
    df.to_csv(temp_file, index=False)

    return FileResponse(
        temp_file,
        media_type="text/csv",
        filename=f"ipd_qma_results_{job_id}.csv"
    )


@app.get("/api/v1/jobs/{job_id}/export/json", tags=["Export"])
async def export_json(job_id: str):
    """Export job results as JSON file."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot export: job status is {job['status']}"
        )

    # Save to temp file
    temp_file = f"/tmp/{job_id}_results.json"
    with open(temp_file, 'w') as f:
        json.dump(job["results"], f, indent=2, default=str)

    return FileResponse(
        temp_file,
        media_type="application/json",
        filename=f"ipd_qma_results_{job_id}.json"
    )


# ========================================================================
# STARTUP AND SHUTDOWN
# ========================================================================

@app.on_event("startup")
async def startup_event():
    """Run on startup."""
    print("=" * 50)
    print("IPD-QMA API Starting...")
    print("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    """Run on shutdown."""
    print("=" * 50)
    print("IPD-QMA API Shutting down...")
    print("=" * 50)


# ========================================================================
# MAIN
# ========================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
