"""A wrapper for getting results from an experiment."""


def to_xid(job):
    assert "-" in job
    return int(job.split("-")[0])


class Checkpoint(object):
    """A single checkpoint of a job."""

    def __init__(self, parent_job, idx):
        self.idx = idx
        self.job = parent_job

    @property
    def relpath(self):
        return "model.ckpt-%i" % self.idx

    # def get_ckpt_dir(self, job):
    @property
    def directory(self):
        return "%s/train" % self.job.root_dir

    @property
    def abspath(self):
        return "%s/%s" % (self.directory, self.relpath)
