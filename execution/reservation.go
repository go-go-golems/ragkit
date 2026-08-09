package execution

import "sync"

type reservation struct {
	mutex    sync.Mutex
	finished bool
	commit   func()
	rollback func()
}

func newReservation(rollback func()) Reservation {
	return newReservationWithCommit(nil, rollback)
}

func newReservationWithCommit(commit, rollback func()) Reservation {
	return &reservation{commit: commit, rollback: rollback}
}

func (reservation *reservation) Commit() {
	reservation.mutex.Lock()
	if reservation.finished {
		reservation.mutex.Unlock()
		return
	}
	reservation.finished = true
	commit := reservation.commit
	reservation.mutex.Unlock()
	if commit != nil {
		commit()
	}
}

func (reservation *reservation) Rollback() {
	reservation.mutex.Lock()
	if reservation.finished {
		reservation.mutex.Unlock()
		return
	}
	reservation.finished = true
	rollback := reservation.rollback
	reservation.mutex.Unlock()
	if rollback != nil {
		rollback()
	}
}
