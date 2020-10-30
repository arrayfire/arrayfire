
#include <async_queue.hpp>
#include <atomic>
#include <thread>

namespace threads {

enum class EventType : int {
    //< The event has been created and can be inserted into the queue
    CREATED,

    /// The event has been inserted into the queue but not has been completed
    QUEUED,

    /// The event has been completed
    COMPLETED,
};

/// A reference counted object that can be used to synchronize points in the
/// async_queue
class event {
    std::shared_ptr<std::atomic<EventType>> event_status;

    void store(EventType val) noexcept {
        event_status->store(val, std::memory_order_release);
    }

   public:
    /// \brief Default constructor. Does not initialize the object. You must use
    /// the
    ///        other constructors or the create function to initialize the event
    event() noexcept : event_status(nullptr) {}
    ~event() noexcept = default;
    explicit event(const event &other) = default;
    event(event &&other) noexcept      = default;

    /// \brief Initializes the event with an integer. This is mostly useful when
    ///        you initialize the event to 0
    explicit event(const int val) : event_status(nullptr) {
        if (val) event_status.reset(new std::atomic<EventType>(EventType(val)));
    }

    /// \brief Initializes the event to a specific event type
    explicit event(const EventType val)
        : event_status(new std::atomic<EventType>(EventType(val))) {}

    /// \brief Set the status of the event
    ///
    /// If the value is non-zero, this function will set the status to the specified
    /// status. If the value is zero, no operation is performed
    ///
    /// \param[in] val The value of the status as an integer
    /// \note This function does not perform checks to see if the value is valid
    event &operator=(int val) noexcept {
        if (event_status) store(static_cast<EventType>(val));
        return *this;
    }

    /// \brief Set the status of the event
    ///
    /// \param[in] Sets the status of the event
    event &operator=(EventType status) noexcept {
        store(status);
        return *this;
    }

    event &operator=(event &&other) noexcept = default;
    event &operator=(event &other) noexcept = default;

    /// \brief Initializes the object. Once created the event can be added to
    ///        the execution queue
    int create() {
        event_status.reset(
            new std::atomic<EventType>(EventType(EventType::CREATED)));
        return 0;
    }

    /// \brief Queues the event to the end of the queue and marks the status to
    ///        QUEUED. Once this point is reached the event will be marked
    ///        COMPLETED
    ///
    /// \param[in] queue The queue that will be marked by this event
    int mark(async_queue &queue) {
        store(EventType::QUEUED);
        queue.enqueue(
            [](std::shared_ptr<std::atomic<EventType>> event_status) noexcept {
                event_status->store(EventType::COMPLETED);
            },
            event_status);
        return 0;
    }

    /// \brief Blocks the queue from progressing until the event has occurred.
    int wait(async_queue &queue) const {
        queue.enqueue(
            [](std::shared_ptr<std::atomic<EventType>> event_status) noexcept {
                while (event_status->load(std::memory_order_acquire) !=
                       EventType::COMPLETED)
                    std::this_thread::yield();
            },
            event_status);
        return 0;
    }

    /// \brief Blocks the current thread from continuing until the event has occurred
    int sync() const noexcept {
        while (event_status->load(std::memory_order_acquire) !=
               EventType::COMPLETED)
            std::this_thread::yield();
        return 0;
    }

    /// \brief Returns true if the event has been created.
    operator bool() const noexcept { return static_cast<bool>(event_status); }
};

}  // namespace threads
