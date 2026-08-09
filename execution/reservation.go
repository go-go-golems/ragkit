package execution

import "sync"

type reservation struct {
	mutex    sync.Mutex
	finished bool
	rollback func()
}

func newReservation(rollback func()) Reservation {
	return &reservation{rollback: rollback}
}

func (reservation *reservation) Commit() {
	reservation.mutex.Lock()
	reservation.finished = true
	reservation.mutex.Unlock()
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
